from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi, HfFolder
import torch

# Load the base model and tokenizer
model_name = "Gunulhona/Gemma-Ko-Merge"
model = AutoModelForCausalLM.from_pretrained(model_name)


# Load the PEFT model
peft_model_path = "/home/conan/workspace/llama3_finetuning/dpo/dpo_output/checkpoint-57500"
peft_model = PeftModel.from_pretrained(
    model=model, 
    model_id=peft_model_path, 
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),# {"load_in_4bit": True},
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Push the model to Hugging Face Hub
repo_name = f"{model_name}-PEFT"
api = HfApi()
token = HfFolder.get_token()

peft_model.push_to_hub(repo_name, use_auth_token=token)
tokenizer.push_to_hub(repo_name, use_auth_token=token)
