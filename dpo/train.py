import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer  # DPOTrainer 사용
from trl import CPOConfig,CPOTrainer  # CPOTrainer 사용
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from data_module import get_dataset
# from unsloth import FastLanguageModel
from accelerate import PartialState

from callbacks import TrainerDebugCallback

# wandb 설정
os.environ["WANDB_PROJECT"]="LLM_DPO"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="0"


# 모델과 토크나이저 불러오기
model_id = "Gunulhona/Gemma-Ko-Merge"
# model_id = "microsoft/Phi-3.5-mini-instruct"
max_seq_len = 4096
batch_per_device = 1
max_epochs = 10
lr = 3e-5 # default 1e-6

print(f'''
--------------------
Model ID: {model_id}
Method : {os.environ.get("RLHF_METHOD")} (default: DPO)
--------------------
''')

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_4bit=False,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.half,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_use_double_quant=True,
        ),
        # device_map="auto",
        device_map={
            "": PartialState().process_index
            # "": torch.cuda.current_device()
            },
        low_cpu_mem_usage=True,
        use_flash_attention_2=False,
        # attn_implementation="eager",
        return_dict=True)
tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True)
model.config.use_cache = False
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_id,
#     trust_remote_code=True,
#     use_cache=True, # Use cache for faster decoding
#     max_seq_length=max_seq_len,
#     dtype=torch.bfloat16, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#     load_in_4bit=False, # Use 4bit quantization to reduce memory usage. Can be False.
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
#     # attn_implementation="flash_attention_2", 
# )

lora_targets=[
        "k_proj",
        # "down_proj",
        # "gate_proj",
        # "q_proj",
        "v_proj",
        # "up_proj",
        # "down_proj",
        # "o_proj"
    ]

# 참조 모델 불러오기 (필요에 따라 수정)
# ref_model = model


# Lora를 기본 모델에 적용
peft_config= LoraConfig( # Lora 설정 정의
    use_mora=True,  # Mora Config
    mora_type=6,
    target_modules=lora_targets,
    r=128,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    # init_lora_weights="gaussian",
    task_type="CAUSAL_LM")

# peft_config = LoHaConfig(
#     target_modules=lora_targets,
#     r=8,
#     alpha=32,
#     task_type="CAUSAL_LM"
# )
model = get_peft_model(
    model=model,
    mixed=False,
    peft_config=peft_config)

# model = FastLanguageModel.get_peft_model(
#     model=model,
#     r=16,
#     max_seq_length=max_seq_len,
#     target_modules=lora_targets,
#     lora_alpha=16,
#     lora_dropout=0, # Dropout = 0 is currently optimized
#     bias="none",    # Bias = "none" is currently optimized
#     use_gradient_checkpointing=True,
#     random_state=3407,
# )
match os.environ.get("RLHF_METHOD", "DPO"):
    case "DPO":
        # DPO 설정 정의
        dpo_config = DPOConfig(
            beta=0.1,
            # loss_type="ipo",
            max_prompt_length=256,
            max_target_length=max_seq_len,
            max_length=max_seq_len,
            learning_rate=lr,
            num_train_epochs=max_epochs,
          # trainer options 
            fp16=True,
            bf16=False,
            tf32=False,
            eval_strategy="steps",
            # fp16_full_eval=True,
            # bf16_full_eval=False,
            output_dir="dpo_output",
            optim="paged_adamw_8bit", # paged_adamw_8bit adamw_bnb_8bit adamw_8bit adamw_hf
            logging_steps=100,
            gradient_accumulation_steps=16,
            generate_during_eval=True,
            dataset_num_proc=8,
            report_to="wandb",
            use_legacy_prediction_loop=True,
            per_device_eval_batch_size=batch_per_device,
            per_device_train_batch_size=batch_per_device,
            per_gpu_eval_batch_size=batch_per_device,
            per_gpu_train_batch_size=batch_per_device,
            lr_scheduler_type="cosine_with_restarts",  #linear, polynomial, reduce_on_plateau, cosine_with_restarts
        )

        # BitsandBytes Paged AdamW Optimizer 설정
        # optimizer_class = bnb.optim.PagedAdamW
        # optimizer_kwargs = {
        #     "lr": dpo_config.learning_rate
        #     }

        dataset = get_dataset(
            dataset_name="Gunulhona/open_dpo_merged",
            tokenizer=tokenizer)

        # DPOTrainer 초기화 (Deepspeed, PagedAdamW 적용)
        rlhf_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            # data_callator=,
            train_dataset=dataset["train"],  # 학습 데이터셋
            eval_dataset=dataset["test"].select(range(50)),  # 학습 데이터셋
            tokenizer=tokenizer,
            # processing_class=tokenizer,
            peft_config=peft_config,
            # deepspeed=deepspeed_config,
            # optimizers=(bnb.optim.PagedAdamW, {"lr": 3e-5}),
            callbacks=[TrainerDebugCallback(model=model, tokenizer=tokenizer)]  # 여러 콜백을 리스트로 전달 가능
        )
    case "SIMPO":
        # CPO 설정 정의
        cpo_config = CPOConfig(
            beta=0.1,
            loss_type="simpo", # SimPO Loss
            cpo_alpha=0.5, # SimPO 학습시 0 으로, CPO-SimPO 학습시 0 이상으로 설정
            max_prompt_length=256,
            learning_rate=lr,
            num_train_epochs=max_epochs,            
          # trainer options 
            max_length=max_seq_len,
            fp16=True,
            bf16=False,
            tf32=False,
            # fp16_full_eval=True,
            # bf16_full_eval=False,
            output_dir="cpo_output",
            optim="paged_adamw_8bit", # paged_adamw_8bit adamw_bnb_8bit adamw_8bit adamw_hf
            logging_steps=100,
            gradient_accumulation_steps=16,
            generate_during_eval=True,
            dataset_num_proc=8,
            report_to="wandb",
            use_legacy_prediction_loop=True,
            per_device_eval_batch_size=batch_per_device,
            per_device_train_batch_size=batch_per_device,
            per_gpu_eval_batch_size=batch_per_device,
            per_gpu_train_batch_size=batch_per_device,
            lr_scheduler_type="cosine_with_restarts",  #linear, polynomial, reduce_on_plateau, cosine_with_restarts
        )

        # BitsandBytes Paged AdamW Optimizer 설정
        # optimizer_class = bnb.optim.PagedAdamW
        # optimizer_kwargs = {
        #     "lr": dpo_config.learning_rate
        #     }

        dataset = get_dataset(
            dataset_name="Gunulhona/open_dpo_merged",
            tokenizer=tokenizer)

        # CPOTrainer 초기화 (Deepspeed, PagedAdamW 적용)
        rlhf_trainer = CPOTrainer(
            model=model,
            # ref_model=None,
            args=cpo_config,
            # data_callator=,
            train_dataset=dataset["train"],  # 학습 데이터셋
            eval_dataset=dataset["test"].select(range(50)),  # 학습 데이터셋
            # tokenizer=tokenizer,
            processing_class=tokenizer,
            peft_config=peft_config,
            # deepspeed=deepspeed_config,
            # optimizers=(bnb.optim.PagedAdamW, {"lr": 3e-5}),
            callbacks=[TrainerDebugCallback(model=model, tokenizer=tokenizer)]  # 여러 콜백을 리스트로 전달 가능
        )


# DPO를 사용하여 모델 학습
if __name__ == "__main__":
    # with torch.autocast("cuda"): 
        rlhf_trainer.train()
