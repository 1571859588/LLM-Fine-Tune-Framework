import torch
from dataclasses import dataclass
from typing import List, Optional
from .base_trainer import BaseTrainer, TrainerConfig
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

@dataclass
class LoRAConfig(TrainerConfig):
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

class LoRATrainer(BaseTrainer):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        config: LoRAConfig,
    ):
        self.setup_lora(model, config)
        super().__init__(model, config)
        
    def setup_lora(self, model: AutoModelForCausalLM, config: LoRAConfig):
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(model, lora_config)
        
    def train(self, train_dataloader):
        self.ds_engine.train()
        
        for epoch in range(self.config.num_epochs):
            for batch in train_dataloader:
                loss = self.ds_engine(batch)
                self.ds_engine.backward(loss)
                self.ds_engine.step()
                
    def evaluate(self, eval_dataloader):
        self.ds_engine.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.ds_engine(batch)
                total_loss += outputs.loss.item()
                
        return total_loss / len(eval_dataloader) 