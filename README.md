# Project Name: llama3_finetuning

## Description:
This project aims to fine-tune the Llama3 model for improved performance in specific tasks.

## Installation:
1. Clone the repository:
```
git clone https://github.com/ppijbb/llama3_finetuning.git
```
2. Install the required dependencies:
```
pip install poetry
poetry lock && poetry install
```

## Usage:
1. Navigate to the project directory:
```
cd llama3_finetuning
```
2. Run the main script:
```
python main.py
```
### SFT with lightning
1. move to lightning directory and install requirements
```bash
cd lightning
pip install -r requirements.txt
```
2. start terminal with tmux(tmux is better than nohup)
```
tmux new -s lightning -d
tmux attach -t lightning
```
3. run trainer
```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export WANDB_API_KEY={ WandB API KEY }
export HF_SECRET_KEY={ Huggingface API KEY }
export HF_DATASETS_CACHE={ Huggingface custom Cache Directory }

huggingface-cli login --token $HF_SECRET_KEY
wandb login --relogin $WANDB_API_KEY

/home/conan/miniconda3/envs/lightning/bin/python\
     trainer.py fit \
        --trainer.fast_dev_run false\
        --trainer.max_epochs 5 \
        --model.learning_rate 3e-3 \
        --data.train_batch_size 4 \
        --data.eval_batch_size 4 
```

### DPO training
1. move to dpo directory
```bash
cd dpo
```
2. start terminal with tmux(tmux is better than nohup)
```
tmux new -s dpo -d
tmux attach -t dpo
```
3. run trainer
```bash
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False'
export TORCH_USE_CUDA_DSA=1
export WANDB_API_KEY={ WandB API KEY }
export HF_SECRET_KEY={ Huggingface API KEY }
export HF_DATASETS_CACHE={ Huggingface custom Cache Directory }

huggingface-cli login --token $HF_SECRET_KEY
wandb login --relogin $WANDB_API_KEY

accelerate launch \
    --config_file "accelerate_config.yaml" \
    train.py
```

## Contributing:
Contributions are welcome! Please follow the guidelines outlined in CONTRIBUTING.md.

## License:
This project is licensed under the MIT License. See LICENSE.md for more details.
