# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from transformers import Seq2SeqTrainer, AutoModelForCausalLM
from torch.nn import functional as F
from ..hidden_divergence import HiddenDivergenceMeter  


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # 加载教师模型
        if self.finetuning_args.teacher_model_name_or_path:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.finetuning_args.teacher_model_name_or_path,
                torch_dtype=torch.float16 if self.args.fp16 else torch.float32
            )
            self.teacher_model.half()
            self.teacher_model.to(self.args.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.teacher_model = None
            
            
        # >>> hidden-probe: 初始化隐藏分歧度量器，并确保两侧输出 hidden_states
        self.hidden_meter = HiddenDivergenceMeter(
            layers=None,   # None = 自动均匀抽层；也可手动如 [2,6,10,14]
            proj_dim=256   # 可设 None 关闭降维；256 能显著省显存/算力
        )

        # 学生模型输出 hidden_states
        if hasattr(self.model, "config"):
            self.model.config.output_hidden_states = True

        # 教师模型输出 hidden_states
        if self.teacher_model is not None and hasattr(self.teacher_model, "config"):
            self.teacher_model.config.output_hidden_states = True
        # <<< hidden-probe
    @override
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # >>> hidden-probe: 学生前向时请求 hidden_states
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # <<< hidden-probe

        logits = outputs.get("logits")
        student_loss = outputs.get("loss")

        if self.teacher_model is not None and self.finetuning_args.kd_ratio > 0:
            with torch.no_grad():
                # >>> hidden-probe: 教师前向也请求 hidden_states
                teacher_outputs = self.teacher_model(**inputs, output_hidden_states=True, return_dict=True)
                # <<< hidden-probe
                teacher_logits = teacher_outputs.get("logits").detach()

            # ====== 原有 KD loss 计算，保持不变 ======
            mask = labels.ne(IGNORE_INDEX).unsqueeze(-1)
            student_logits = logits.float()
            teacher_logits = teacher_logits.float()
            masked_student_logits = torch.masked_select(student_logits, mask).view(-1, student_logits.size(-1))
            masked_teacher_logits = torch.masked_select(teacher_logits, mask).view(-1, teacher_logits.size(-1))
            student_log_probs = F.log_softmax(masked_student_logits, dim=-1)
            teacher_probs = F.softmax(masked_teacher_logits, dim=-1)
            kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            alpha = self.finetuning_args.kd_ratio
            loss = (1 - alpha) * student_loss + alpha * kd_loss
            logger.info(f"CE loss: {student_loss.detach().item()}, KL loss: {kd_loss.detach().item()}")

            # >>> hidden-probe: 计算 D（每步）
            try:
                attn_mask = inputs.get("attention_mask")
                if attn_mask is None:
                    attn_mask = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=logits.device)

                meter_out = self.hidden_meter(
                    hS=list(outputs.hidden_states),           # Student: List[Tensor [B,T,Ds]]
                    hT=list(teacher_outputs.hidden_states),   # Teacher: List[Tensor [B,T,Dt]]
                    attn_mask=attn_mask,
                    use_last_token=False                      # True 用最后 token，False 用 mask-mean
                )

                # 通过 HF Trainer 的统一接口写日志（W&B/TensorBoard/MLflow 会自动接收）
                self.log({
                    "hidden/divergence": meter_out["D"].item(),
                    "hidden/cka_mean": meter_out["cka_mean"].item(),
                    "hidden/nmse_mean": meter_out["nmse_mean"].item(),
                })
            except Exception as e:
                logger.info(f"[hidden-probe] skipped due to: {e}")
            # <<< hidden-probe

        else:
            loss = student_loss
            logger.info(f"CE loss: {student_loss.detach().item()}")

        return (loss, outputs) if return_outputs else loss

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
