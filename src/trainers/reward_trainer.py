import torch
from typing import Dict, Tuple
from .base_trainer import BaseTrainer
from .config import RewardModelConfig
from transformers import AutoModelForSequenceClassification

class RewardTrainer(BaseTrainer):
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        config: RewardModelConfig,
    ):
        super().__init__(model, config)
        
    def compute_reward_loss(self, chosen_outputs, rejected_outputs):
        chosen_rewards = self.ds_engine(chosen_outputs).logits
        rejected_rewards = self.ds_engine(rejected_outputs).logits
        
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return loss
        
    def train(self, train_dataloader):
        self.ds_engine.train()
        
        for epoch in range(self.config.num_epochs):
            for batch in train_dataloader:
                chosen_outputs = {
                    'input_ids': batch['chosen_input_ids'],
                    'attention_mask': batch['chosen_attention_mask']
                }
                rejected_outputs = {
                    'input_ids': batch['rejected_input_ids'],
                    'attention_mask': batch['rejected_attention_mask']
                }
                
                loss = self.compute_reward_loss(chosen_outputs, rejected_outputs)
                self.ds_engine.backward(loss)
                self.ds_engine.step() 