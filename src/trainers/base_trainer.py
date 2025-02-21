from abc import ABC, abstractmethod
import torch
import deepspeed
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class TrainerConfig:
    model_name: str
    batch_size: int
    learning_rate: float = 5e-5
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    fp16: bool = True
    local_rank: int = -1
    deepspeed_config: Optional[Dict] = None

class BaseTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
    ):
        self.model = model
        self.config = config
        self.ds_engine = None
        self.setup_deepspeed()
        
    def setup_deepspeed(self):
        # Default DeepSpeed configuration if not provided
        if self.config.deepspeed_config is None:
            self.config.deepspeed_config = {
                "train_batch_size": self.config.batch_size,
                "fp16": {
                    "enabled": self.config.fp16
                },
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.config.learning_rate,
                        "weight_decay": self.config.weight_decay
                    }
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": self.config.learning_rate,
                        "warmup_num_steps": self.config.warmup_steps
                    }
                },
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu"
                    }
                }
            }
        
        # Initialize DeepSpeed
        self.ds_engine, _, _, _ = deepspeed.initialize(
            model=self.model,
            config=self.config.deepspeed_config,
            model_parameters=self.model.parameters()
        )
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
    
    def save_checkpoint(self, path: str):
        self.ds_engine.save_checkpoint(path)
    
    def load_checkpoint(self, path: str):
        _, client_state = self.ds_engine.load_checkpoint(path)
        return client_state 