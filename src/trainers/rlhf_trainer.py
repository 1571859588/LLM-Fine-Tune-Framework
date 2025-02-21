import torch
from typing import Dict, Tuple
from .base_trainer import BaseTrainer
from .config import RLHFConfig
from transformers import AutoModelForCausalLM
from .reward_trainer import RewardTrainer

class RLHFTrainer(BaseTrainer):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        reward_model: RewardTrainer,
        config: RLHFConfig,
    ):
        super().__init__(model, config)
        self.reward_model = reward_model
        
    def compute_ppo_loss(self, batch, old_logprobs):
        logits = self.ds_engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        ).logits
        
        new_logprobs = torch.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            rewards = self.reward_model(batch["input_ids"], batch["attention_mask"]).logits
            
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * rewards
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
        
    def train(self, train_dataloader):
        self.ds_engine.train()
        
        for epoch in range(self.config.num_epochs):
            for batch in train_dataloader:
                with torch.no_grad():
                    old_outputs = self.ds_engine(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                    old_logprobs = torch.log_softmax(old_outputs.logits, dim=-1)
                
                for _ in range(self.config.ppo_epochs):
                    loss = self.compute_ppo_loss(batch, old_logprobs)
                    self.ds_engine.backward(loss)
                    self.ds_engine.step() 