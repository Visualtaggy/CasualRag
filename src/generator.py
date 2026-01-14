# File: src/generator.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_best_device():
    """
    Automatically selects the best available hardware.
    Priority: CUDA (Nvidia) > MPS (Mac) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class CausalGenerator:
    # Change default to a better model that runs on Mac
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", device=None):
        
        if device is None:
            self.device = get_best_device()
        else:
            self.device = device
            
        print(f"Loading Generator: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with trust_remote_code=True for Phi-3
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True 
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt, max_new_tokens=100):
        # Ensure inputs are on the same device as the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def get_logits(self, context, answer):
        """
        Computes logits specifically for the ANSWER tokens, masking out the context.
        This ensures that even if Context A is longer than Context B,
        we only compare the distributions for the Answer tokens.
        """
        # 1. Tokenize the Prompt (Context + Question part)
        # We need to know where the prompt ends and the answer begins.
        prompt_text = f"{context}\nAnswer: "
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = prompt_inputs.input_ids.shape[1]

        # 2. Tokenize the Full Sequence (Prompt + Answer)
        full_text = f"{context}\nAnswer: {answer}"
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        # 3. Run the model
        with torch.no_grad():
            outputs = self.model(**full_inputs)
        
        # 4. Smart Slicing
        # The model predicts token[i+1] based on logits[i].
        # We want the predictions for the Answer tokens.
        # The first token of the answer is predicted by the last token of the prompt.
        # So we slice from (prompt_len - 1) to the end.
        
        # Calculate how many tokens are in the answer part
        seq_len = full_inputs.input_ids.shape[1]
        
        # Slice: Start at last prompt token -> End at last answer token
        start_idx = prompt_len - 1
        end_idx = seq_len - 1
        
        answer_logits = outputs.logits[:, start_idx:end_idx, :]
        
        return answer_logits