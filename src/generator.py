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
    # Set device=None by default to trigger auto-detection
    def __init__(self, model_name="gpt2", device=None):
        
        # 1. Auto-detect device if none is provided
        if device is None:
            self.device = get_best_device()
        else:
            self.device = device
            
        print(f"Loading Generator: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 2. Load model
        # Note: We use float16 on GPU for speed, but float32 on CPU for stability
        torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype
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
        full_text = f"{context}\nAnswer: {answer}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.logits