import os
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from src.trainers.config import (
    BaseConfig, LoRAConfig, RLHFConfig, RewardModelConfig, 
    DPOConfig, SimPOConfig, KTOConfig
)
from src.trainers.lora_trainer import LoRATrainer
from src.trainers.reward_trainer import RewardTrainer
from src.trainers.dpo_trainer import DPOTrainer
from src.trainers.rlhf_trainer import RLHFTrainer
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }
    

def create_trainer(training_type: str, config_path: str):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    if training_type == "lora":
        config = LoRAConfig(**config_dict)
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        return LoRATrainer(model, config)
        
    elif training_type == "reward":
        config = RewardModelConfig(**config_dict)
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        return RewardTrainer(model, config)
        
    elif training_type == "dpo":
        config = DPOConfig(**config_dict)
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        reference_model = AutoModelForCausalLM.from_pretrained(config.reference_model_name)
        return DPOTrainer(model, reference_model, config)
        
    elif training_type == "rlhf":
        config = RLHFConfig(**config_dict)
        model = AutoModelForCausalLM.from_pretrained(config.model_name)
        reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_name)
        reward_trainer = RewardTrainer(reward_model, RewardModelConfig(**config_dict))
        return RLHFTrainer(model, reward_trainer, config)

def create_dataloader(training_type: str, config: BaseConfig) -> DataLoader:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    if training_type == "lora":
        dataset = TextDataset(texts=load_training_data(), tokenizer=tokenizer, max_length=config.max_length)
    elif training_type in ["reward", "dpo"]:
        dataset = PreferenceDataset(
            chosen_texts=load_preference_data()["chosen"],
            rejected_texts=load_preference_data()["rejected"],
            tokenizer=tokenizer,
            max_length=config.max_length
        )
    elif training_type == "rlhf":
        dataset = RLHFDataset(
            prompts=load_rlhf_data()["prompts"],
            responses=load_rlhf_data()["responses"],
            tokenizer=tokenizer,
            max_length=config.max_length
        )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )

class PreferenceDataset(Dataset):
    def __init__(self, chosen_texts, rejected_texts, tokenizer, max_length=512):
        self.chosen_texts = chosen_texts
        self.rejected_texts = rejected_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.chosen_texts)

    def __getitem__(self, idx):
        chosen_encodings = self.tokenizer(
            self.chosen_texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        rejected_encodings = self.tokenizer(
            self.rejected_texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'chosen_input_ids': chosen_encodings['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encodings['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encodings['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encodings['attention_mask'].squeeze()
        }

class RLHFDataset(Dataset):
    def __init__(self, prompts, responses, tokenizer, max_length=512):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.prompts[idx],
            self.responses[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }

def load_training_data():
    # Implement your data loading logic here
    # For example:
    return ["Sample text 1", "Sample text 2"]  # Replace with actual data

def load_preference_data():
    return {
        "chosen": ["Good response 1", "Good response 2"],
        "rejected": ["Bad response 1", "Bad response 2"]
    }

def load_rlhf_data():
    return {
        "prompts": ["Prompt 1", "Prompt 2"],
        "responses": ["Response 1", "Response 2"]
    }

def main():
    training_type = os.environ.get("TRAINING_TYPE", "lora")
    config_path = f"configs/{training_type}_config.yaml"
    
    trainer = create_trainer(training_type, config_path)
    
    # Create dataset and dataloader based on training type
    train_dataloader = create_dataloader(training_type, trainer.config)
    
    # Train the model
    trainer.train(train_dataloader)
    
    # Save the model
    trainer.save_checkpoint(f"checkpoints/{training_type}_model")

if __name__ == "__main__":
    main() 