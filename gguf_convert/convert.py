from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Gunulhona/Gemma-Ko-Merge-PEFT",
    dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit = True,
    )
model.save_pretrained_gguf(
    "/home/work/workspace/doc_gen/sample/gguf", 
    tokenizer, 
    quantization_method="q4_k_m"
    )
