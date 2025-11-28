import torch
from opencompass.models import HuggingFaceCausalLM
from opencompass.registry import MODELS  # <--- [关键] 必须引入注册表
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

class ProxyLogitsProcessor(LogitsProcessor):
    def __init__(self, expert_model, anti_expert_model, base_tokenizer):
        self.expert = expert_model
        self.anti_expert = anti_expert_model
        self.tokenizer = base_tokenizer
        self.expert_past_key_values = None
        self.anti_expert_past_key_values = None
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 确保 expert 和 anti-expert 处于 eval 模式
        self.expert.eval()
        self.anti_expert.eval()

        # 获取当前序列长度
        seq_len = input_ids.shape[1]
        
        with torch.no_grad():
            # 1. 处理 Expert 模型 (A+)
            if self.expert_past_key_values is None:
                e_inputs = input_ids
            else:
                e_inputs = input_ids[:, -1:]
            
            e_outputs = self.expert(
                e_inputs, 
                past_key_values=self.expert_past_key_values,
                use_cache=True
            )
            self.expert_past_key_values = e_outputs.past_key_values
            # 取最后一个token的logits
            e_logits = e_outputs.logits[:, -1, :] 

            # 2. 处理 Anti-Expert 模型 (A)
            if self.anti_expert_past_key_values is None:
                a_inputs = input_ids
            else:
                a_inputs = input_ids[:, -1:]
                
            a_outputs = self.anti_expert(
                a_inputs, 
                past_key_values=self.anti_expert_past_key_values,
                use_cache=True
            )
            self.anti_expert_past_key_values = a_outputs.past_key_values
            a_logits = a_outputs.logits[:, -1, :]

        # 3. Proxy Tuning 核心公式: 
        # Logits_final = Logits_Base + (Logits_Expert - Logits_AntiExpert)
        # scores 即为 Base 模型的当前 logits
        proxy_logits = scores + (e_logits - a_logits)
        
        return proxy_logits

# <--- [关键] 添加这个装饰器，将类注册到 OpenCompass 的 MODELS 域中
@MODELS.register_module() 
class ProxyTuningQwen(HuggingFaceCausalLM):
    def __init__(self, 
                 expert_path, 
                 anti_expert_path, 
                 **kwargs):
        super().__init__(**kwargs)
        self.expert_path = expert_path
        self.anti_expert_path = anti_expert_path
        self.expert_model = None
        self.anti_expert_model = None

    def _load_model(self):
        # 1. 加载 Base Model (B)
        super()._load_model()
        
        # 2. 加载 Expert (A+) 和 Anti-Expert (A)
        # 强制与 Base Model 使用同一张卡
        device = self.model.device
        dtype = self.model.dtype 
        
        print(f"Loading Expert from {self.expert_path}...")
        self.expert_model = AutoModelForCausalLM.from_pretrained(
            self.expert_path, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            device_map=device
        ).eval()

        print(f"Loading Anti-Expert from {self.anti_expert_path}...")
        self.anti_expert_model = AutoModelForCausalLM.from_pretrained(
            self.anti_expert_path, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            device_map=device
        ).eval()

    def generate(self, inputs, max_out_len, **kwargs):
        # 实例化 Proxy Processor
        proxy_processor = ProxyLogitsProcessor(
            self.expert_model,
            self.anti_expert_model,
            self.tokenizer
        )
        
        # 注入 LogitsProcessor
        if 'logits_processor' not in kwargs:
            kwargs['logits_processor'] = LogitsProcessorList()
        
        kwargs['logits_processor'].append(proxy_processor)

        return super().generate(inputs, max_out_len, **kwargs)
