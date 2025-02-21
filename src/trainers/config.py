from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union

@dataclass
class BaseConfig:
    model_name: str
    batch_size: int
    learning_rate: float = 5e-5
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    fp16: bool = True
    local_rank: int = -1
    max_length: int = 512
    deepspeed_config: Optional[Dict] = None

@dataclass
class LoRAConfig(BaseConfig):
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

@dataclass
class RLHFConfig(BaseConfig):
    ppo_epochs: int = 4
    kl_penalty: float = 0.1
    value_loss_coef: float = 0.1
    entropy_coef: float = 0.01
    clip_epsilon: float = 0.2
    reward_model_name: str = None

@dataclass
class RewardModelConfig(BaseConfig):
    num_comparisons: int = 1000
    margin_loss: float = 0.5
    temperature: float = 1.0

@dataclass
class DPOConfig(BaseConfig):
    reference_model_name: str = None
    beta: float = 0.1
    max_prompt_length: int = 256
    max_response_length: int = 256

@dataclass
class SimPOConfig(BaseConfig):
    reference_model_name: str = None
    similarity_threshold: float = 0.7
    contrastive_loss_weight: float = 0.5

@dataclass
class KTOConfig(BaseConfig):
    reference_model_name: str = None
    knowledge_temperature: float = 0.1
    distillation_alpha: float = 0.5 