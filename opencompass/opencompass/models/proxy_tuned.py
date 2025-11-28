# opencompass/models/proxy_tuned.py
"""
Proxy-Tuning Model Implementation
基于论文: Tuning Language Models by Proxy (Liu et al., 2024)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger


@MODELS.register_module()
class ProxyTunedHuggingFace(BaseModel):
    """Proxy-Tuning 模型"""
    
    def __init__(
        self,
        base_model_path: str,
        expert_model_path: str,
        anti_expert_model_path: str,
        tokenizer_path: Optional[str] = None,
        alpha: float = 1.0,
        max_seq_len: int = 4096,
        max_out_len: int = 4096,
        batch_size: int = 1,
        tokenizer_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
        base_device: str = "cuda:0",
        expert_device: str = "cuda:2",
        anti_expert_device: str = "cuda:4",
        use_auto_device_map: bool = False,
        meta_template: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs
    ):
        # 不调用父类 __init__ 中的复杂逻辑，手动设置必要属性
        self.path = base_model_path
        self.max_seq_len = max_seq_len
        self.template_parser = None
        self.meta_template = meta_template
        self.logger = get_logger()
        
        self.base_model_path = base_model_path
        self.expert_model_path = expert_model_path
        self.anti_expert_model_path = anti_expert_model_path
        self.tokenizer_path = tokenizer_path or base_model_path
        self.alpha = alpha
        self.max_out_len = max_out_len
        self.batch_size = batch_size
        
        self.base_device = base_device
        self.expert_device = expert_device
        self.anti_expert_device = anti_expert_device
        self.use_auto_device_map = use_auto_device_map
        
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer_kwargs.setdefault('padding_side', 'left')
        self.tokenizer_kwargs.setdefault('trust_remote_code', True)
        
        self.model_kwargs = model_kwargs or {}
        self.model_kwargs.setdefault('trust_remote_code', True)
        # 不在这里设置 torch_dtype，在加载时设置
        
        self.generation_kwargs = generation_kwargs or {}
        
        self._loaded = False
        self.tokenizer = None
        self.base_model = None
        self.expert_model = None
        self.anti_expert_model = None
        
    def _load_models(self):
        if self._loaded:
            return
        
        # 延迟导入
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.logger.info(f'[Proxy-Tuning] Loading tokenizer from {self.tokenizer_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, **self.tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 设置 dtype
        load_kwargs = dict(self.model_kwargs)
        if 'torch_dtype' not in load_kwargs:
            load_kwargs['torch_dtype'] = torch.bfloat16
        elif isinstance(load_kwargs['torch_dtype'], str):
            dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
            load_kwargs['torch_dtype'] = dtype_map.get(load_kwargs['torch_dtype'], torch.bfloat16)
        
        self.logger.info(f'[Proxy-Tuning] Loading base model (M) from {self.base_model_path}')
        if self.use_auto_device_map:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, device_map='auto', **load_kwargs
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, **load_kwargs
            ).to(self.base_device)
        self.base_model.eval()
        
        self.logger.info(f'[Proxy-Tuning] Loading expert model (M+) from {self.expert_model_path}')
        if self.use_auto_device_map:
            self.expert_model = AutoModelForCausalLM.from_pretrained(
                self.expert_model_path, device_map='auto', **load_kwargs
            )
        else:
            self.expert_model = AutoModelForCausalLM.from_pretrained(
                self.expert_model_path, **load_kwargs
            ).to(self.expert_device)
        self.expert_model.eval()
        
        self.logger.info(f'[Proxy-Tuning] Loading anti-expert model (M-) from {self.anti_expert_model_path}')
        if self.use_auto_device_map:
            self.anti_expert_model = AutoModelForCausalLM.from_pretrained(
                self.anti_expert_model_path, device_map='auto', **load_kwargs
            )
        else:
            self.anti_expert_model = AutoModelForCausalLM.from_pretrained(
                self.anti_expert_model_path, **load_kwargs
            ).to(self.anti_expert_device)
        self.anti_expert_model.eval()
        
        self._loaded = True
        self.logger.info('[Proxy-Tuning] All models loaded!')
        
    def get_token_len(self, prompt: str) -> int:
        self._load_models()
        return len(self.tokenizer.encode(prompt))
    
    def _get_device(self, model):
        return next(model.parameters()).device
    
    def _proxy_logits(self, input_ids, attention_mask):
        """计算 proxy-tuned logits: s_M + α × (s_M+ - s_M-)"""
        with torch.no_grad():
            base_dev = self._get_device(self.base_model)
            base_out = self.base_model(
                input_ids.to(base_dev),
                attention_mask=attention_mask.to(base_dev),
                return_dict=True
            )
            base_logits = base_out.logits
            
            expert_dev = self._get_device(self.expert_model)
            expert_out = self.expert_model(
                input_ids.to(expert_dev),
                attention_mask=attention_mask.to(expert_dev),
                return_dict=True
            )
            expert_logits = expert_out.logits.to(base_dev)
            
            anti_dev = self._get_device(self.anti_expert_model)
            anti_out = self.anti_expert_model(
                input_ids.to(anti_dev),
                attention_mask=attention_mask.to(anti_dev),
                return_dict=True
            )
            anti_logits = anti_out.logits.to(base_dev)
            
            # Proxy-tuning 公式
            proxy_logits = base_logits + self.alpha * (expert_logits - anti_logits)
            
        return proxy_logits
    
    def _generate_proxy(self, input_ids, attention_mask, max_new_tokens, 
                        temperature=1.0, top_k=1, top_p=0.9):
        """逐token生成"""
        import torch.nn.functional as F
        
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else eos_id
        device = self._get_device(self.base_model)
        batch_size = input_ids.shape[0]
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_new_tokens):
            logits = self._proxy_logits(input_ids, attention_mask)
            next_logits = logits[:, -1, :]
            
            # 温度调整
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Top-k 采样
            if top_k > 0 and top_k < next_logits.size(-1):
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # 贪婪解码 (top_k=1)
            if top_k == 1:
                next_token = torch.argmax(next_logits, dim=-1)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # 检查 EOS
            eos_reached = eos_reached | (next_token == eos_id)
            next_token = torch.where(eos_reached, torch.tensor(pad_id, device=device), next_token)
            
            # 更新序列
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            new_mask = (~eos_reached).long().unsqueeze(-1).to(attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=-1)
            
            if eos_reached.all():
                break
                
            # 打印进度
            if (step + 1) % 100 == 0:
                self.logger.info(f'[Proxy-Tuning] Generated {step + 1}/{max_new_tokens} tokens')
                
        return input_ids
    
    def generate(self, inputs: List[str], max_out_len: int = 512, **kwargs) -> List[str]:
        """OpenCompass generate 接口"""
        self._load_models()
        
        # 合并生成参数
        gen_kwargs = dict(self.generation_kwargs)
        gen_kwargs.update(kwargs)
        temperature = gen_kwargs.get('temperature', 0)
        top_k = gen_kwargs.get('top_k', 1)
        top_p = gen_kwargs.get('top_p', 0.9)
        
        results = []
        total = len(inputs)
        
        for idx, prompt in enumerate(inputs):
            self.logger.info(f'[Proxy-Tuning] Processing {idx+1}/{total}')
            
            try:
                enc = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_seq_len,
                    padding=False
                )
                input_len = enc['input_ids'].shape[1]
                
                out_ids = self._generate_proxy(
                    enc['input_ids'],
                    enc['attention_mask'],
                    max_new_tokens=max_out_len,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_k=top_k,
                    top_p=top_p
                )
                
                gen_ids = out_ids[0, input_len:]
                # 移除 pad tokens
                if self.tokenizer.pad_token_id is not None:
                    gen_ids = gen_ids[gen_ids != self.tokenizer.pad_token_id]
                # 移除 eos token
                if self.tokenizer.eos_token_id is not None:
                    gen_ids = gen_ids[gen_ids != self.tokenizer.eos_token_id]
                    
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                results.append(text)
                
            except Exception as e:
                self.logger.error(f'[Proxy-Tuning] Error processing prompt {idx}: {e}')
                results.append('')
                
        return results
    
    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """计算 PPL"""
        self._load_models()
        
        ppls = []
        for idx, text in enumerate(inputs):
            try:
                enc = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_seq_len
                )
                
                logits = self._proxy_logits(enc['input_ids'], enc['attention_mask'])
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = enc['input_ids'][..., 1:].contiguous().to(shift_logits.device)
                
                if mask_length and idx < len(mask_length) and mask_length[idx] > 0:
                    shift_logits = shift_logits[..., -mask_length[idx]:, :]
                    shift_labels = shift_labels[..., -mask_length[idx]:]
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
                ppls.append(torch.exp(loss).item())
                
            except Exception as e:
                self.logger.error(f'[Proxy-Tuning] PPL error: {e}')
                ppls.append(float('inf'))
                
        return ppls
