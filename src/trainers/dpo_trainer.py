import torch
from typing import Dict
from .base_trainer import BaseTrainer
from .config import DPOConfig
from transformers import AutoModelForCausalLM

class DPOTrainer(BaseTrainer):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        reference_model: AutoModelForCausalLM,
        config: DPOConfig,
    ):
        super().__init__(model, config)
        self.reference_model = reference_model
        
    def compute_dpo_loss(self, batch):
        policy_chosen_logps = self.ds_engine(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"]
        ).logits
        
        policy_rejected_logps = self.ds_engine(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"]
        ).logits
        
        with torch.no_grad():
            reference_chosen_logps = self.reference_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"]
            ).logits
            
            reference_rejected_logps = self.reference_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"]
            ).logits
        
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps
        
        loss = -torch.log(torch.sigmoid(self.config.beta * (chosen_rewards - rejected_rewards))).mean()
        return loss
        
    def train(self, train_dataloader):
        self.ds_engine.train()
        
        for epoch in range(self.config.num_epochs):
            for batch in train_dataloader:
                loss = self.compute_dpo_loss(batch)
                self.ds_engine.backward(loss)
                self.ds_engine.step() 