import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM

def load_model(model_path, peft_path=None):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
    peft_path,
    torch_dtype=torch.bfloat16,
    quantization_config= {"load_in_8bit": True},
    device_map="auto"
    )
        
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Example usage
    base_model_path = "Gunulhona/Gemma-System-9B"  # e.g. "mistralai/Mistral-7B-v0.1"
    peft_path = "/home/conan/workspace/llama3_finetuning/dpo/cpo_output/checkpoint-23500"  # Path to your DPO finetuned adapter
    
    # Load model
    model, tokenizer = load_model(base_model_path, peft_path)
    
    # Generate text
    prompt = "Write a story about a space adventure:"
    print(f"Prompt: {prompt}\n")
    response = generate_text(prompt, model, tokenizer)
    print(f"Generated Response: {response}")
