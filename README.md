# LLM Fine-tuning Framework

A comprehensive framework for fine-tuning Large Language Models using various techniques including LoRA, RLHF, DPO, and more.

## Features

- Multiple training methods supported:
  - LoRA (Low-Rank Adaptation)
  - RLHF (Reinforcement Learning from Human Feedback)
  - DPO (Direct Preference Optimization)
  - Reward Modeling
  - SimPO (Similarity-based Preference Optimization)
  - KTO (Knowledge Transfer Optimization)
- DeepSpeed integration for efficient training
- Configurable training parameters via YAML files
- Modular architecture for easy extension

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/llm-fine-tuning-framework.git
cd llm-fine-tuning-framework
```

2. Install dependencies:
bash
pip install -r requirements.txt

## Usage

1. Configure your training parameters in `configs/{method}_config.yaml`

2. Run training:


```bash
export TRAINING_TYPE=lora # or reward, dpo, rlhf
python main.py
```

## Configuration

Each training method has its own configuration file in the `configs/` directory:

- `lora_config.yaml`: LoRA training parameters
- `reward_config.yaml`: Reward model training parameters
- `dpo_config.yaml`: DPO training parameters
- `rlhf_config.yaml`: RLHF training parameters

## Project Structure

```
├── configs/
│ ├── lora_config.yaml
│ ├── reward_config.yaml
│ ├── dpo_config.yaml
│ └── rlhf_config.yaml
├── src/
│ └── trainers/
│ ├── base_trainer.py
│ ├── lora_trainer.py
│ ├── reward_trainer.py
│ ├── dpo_trainer.py
│ └── rlhf_trainer.py
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- DeepSpeed 0.10+
- PEFT 0.5+

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
