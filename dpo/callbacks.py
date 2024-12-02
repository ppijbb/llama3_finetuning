from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import torch

# 커스텀 콜백 정의
class TrainerDebugCallback(TrainerCallback):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def on_step_begin(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kwargs):
        # print(f"Starting step {state.global_step}")
        pass
    
    def on_step_end(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kwargs):
        # print(f"Finished step {state.global_step}")
        # test_input = [{"role": "user", "content": "What is the capital of France?"}]
        # with torch.inference_mode():
        #     print(
        #         self.tokenizer.decode(
        #             self.model.generate(
        #                 input_ids=self.tokenizer.apply_chat_template(
        #                     test_input, 
        #                     tokenize=True, 
        #                     add_generation_prompt=True,
        #                     return_tensors="pt"),
        #                 do_sample=True, 
        #                 temperature=0.3,
        #                 max_length=128
        #                 )[0],
        #             skip_special_tokens=True
        #             )
        #         )
        pass
    
    def on_epoch_begin(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kwargs):
        # print(f"Starting epoch {state.epoch}")
        pass
        
    def on_epoch_end(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kwargs):
        # print(f"Finished epoch {state.epoch}")
        pass
        
    def on_log(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
            
    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        test_input = [{"role": "user", "content": "What is the capital of France?"}]
        with torch.inference_mode():
            print(
                kwargs["processing_class"].decode(
                    kwargs["model"].generate(
                        input_ids=kwargs["processing_class"].apply_chat_template(
                            test_input, 
                            tokenize=True, 
                            add_generation_prompt=True,
                            return_tensors="pt").to(kwargs["model"].device),
                        do_sample=True, 
                        temperature=0.3,
                        max_length=128
                        )[0],
                    skip_special_tokens=True
                    )
                )
