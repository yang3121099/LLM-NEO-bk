import torch
from opencompass.models import HuggingFaceCausalLM
from opencompass.registry import MODELS
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

class ProxyLogitsProcessor(LogitsProcessor):
    def __init__(self, expert_model, anti_expert_model):
        self.expert = expert_model
        self.anti_expert = anti_expert_model
        self.expert_past_key_values = None
        self.anti_expert_past_key_values = None
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.expert.eval()
        self.anti_expert.eval()

        with torch.no_grad():
            # 利用 KV Cache 加速
            if self.expert_past_key_values is None:
                curr_inputs = input_ids 
            else:
                curr_inputs = input_ids[:, -1:] 
            
            # 1. Expert (A+)
            e_outputs = self.expert(
                curr_inputs, 
                past_key_values=self.expert_past_key_values,
                use_cache=True
            )
            self.expert_past_key_values = e_outputs.past_key_values
            e_logits = e_outputs.logits[:, -1, :] 

            # 2. Anti-Expert (A)
            a_outputs = self.anti_expert(
                curr_inputs, 
                past_key_values=self.anti_expert_past_key_values,
                use_cache=True
            )
            self.anti_expert_past_key_values = a_outputs.past_key_values
            a_logits = a_outputs.logits[:, -1, :]

        # 3. Apply Offset
        proxy_logits = scores + (e_logits - a_logits)
        
        return proxy_logits

@MODELS.register_module() 
class ProxyQwen(HuggingFaceCausalLM):
    def __init__(self, 
                 expert_path, 
                 anti_expert_path, 
                 **kwargs):
        # =========== [关键修改点] 先赋值，再初始化父类 ===========
        self.expert_path = expert_path
        self.anti_expert_path = anti_expert_path
        self.expert_model = None
        self.anti_expert_model = None
        
        # 调用父类初始化，父类会立即调用 _load_model
        # 此时 self.expert_path 已经存在，不会报错了
        super().__init__(**kwargs)

    def _load_model(self, **kwargs):
        # 1. 加载 Base Model
        super()._load_model(**kwargs)
        
        # 2. 加载 Expert 和 Anti-Expert
        if self.model is not None:
            device = self.model.device
            dtype = self.model.dtype 
        else:
            device = "cuda"
            dtype = torch.float16
        
        print(f"[ProxyQwen] Loading Expert from {self.expert_path}...")
        self.expert_model = AutoModelForCausalLM.from_pretrained(
            self.expert_path, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            device_map=device 
        ).eval()

        print(f"[ProxyQwen] Loading Anti-Expert from {self.anti_expert_path}...")
        self.anti_expert_model = AutoModelForCausalLM.from_pretrained(
            self.anti_expert_path, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            device_map=device
        ).eval()

    def generate(self, inputs, max_out_len, **kwargs):
        proxy_processor = ProxyLogitsProcessor(
            self.expert_model,
            self.anti_expert_model
        )
        
        if 'logits_processor' not in kwargs:
            kwargs['logits_processor'] = LogitsProcessorList()
        kwargs['logits_processor'].append(proxy_processor)

        return super().generate(inputs, max_out_len, **kwargs)
