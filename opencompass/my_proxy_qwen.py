import torch
from transformers import LogitsProcessor, AutoModelForCausalLM
from opencompass.models import HuggingFaceCausalLM
# 必须导入注册表
from opencompass.registry import MODELS

class ProxyTuningLogitsProcessor(LogitsProcessor):
    def __init__(self, expert, anti_expert, alpha=1.0):
        self.expert = expert
        self.anti_expert = anti_expert
        self.alpha = alpha
        self.expert_past_key_values = None
        self.anti_expert_past_key_values = None
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.expert_past_key_values is None:
            curr_inputs = input_ids
        else:
            curr_inputs = input_ids[:, -1:]

        with torch.no_grad():
            expert_out = self.expert(
                curr_inputs,
                past_key_values=self.expert_past_key_values,
                use_cache=True
            )
            self.expert_past_key_values = expert_out.past_key_values
            expert_logits = expert_out.logits[:, -1, :]

        with torch.no_grad():
            anti_out = self.anti_expert(
                curr_inputs,
                past_key_values=self.anti_expert_past_key_values,
                use_cache=True
            )
            self.anti_expert_past_key_values = anti_out.past_key_values
            anti_logits = anti_out.logits[:, -1, :]

        offset = self.alpha * (expert_logits - anti_logits)
        final_scores = scores + offset.to(scores.device)
        return final_scores

# === 注册模块 ===
@MODELS.register_module()
class ProxyQwen(HuggingFaceCausalLM):
    def __init__(self, 
                 expert_path: str, 
                 anti_expert_path: str, 
                 alpha: float = 1.0, 
                 **kwargs):
        super().__init__(**kwargs)
        self.expert_path = expert_path
        self.anti_expert_path = anti_expert_path
        self.alpha = alpha
        
        print(f"Loading Expert from: {expert_path}")
        self.expert_model = self._load_aux_model(expert_path)
        print(f"Loading Anti-Expert from: {anti_expert_path}")
        self.anti_expert_model = self._load_aux_model(anti_expert_path)
        
        self.expert_model.eval()
        self.anti_expert_model.eval()

    def _load_aux_model(self, path):
        return AutoModelForCausalLM.from_pretrained(
            path,
            device_map=self.model.device, 
            trust_remote_code=True,
            torch_dtype=self.model.dtype
        )

    def _generate(self, inputs, max_length, **kwargs):
        processor = ProxyTuningLogitsProcessor(
            expert=self.expert_model,
            anti_expert=self.anti_expert_model,
            alpha=self.alpha
        )
        if 'logits_processor' not in kwargs:
            from transformers import LogitsProcessorList
            kwargs['logits_processor'] = LogitsProcessorList()
        
        if isinstance(kwargs['logits_processor'], list):
            kwargs['logits_processor'].append(processor)
        else:
            kwargs['logits_processor'].append(processor)

        return super()._generate(inputs, max_length, **kwargs)
