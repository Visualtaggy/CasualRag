# src/entailment.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EntailmentGrader:
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading NLI Judge: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def check_entailment(self, premise, hypothesis):
        """
        Returns probabilities for [Contradiction, Neutral, Entailment]
        """
        input_text = f"{premise} </s></s> {hypothesis}" # BART uses </s></s> separator
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # BART-Large-MNLI output logits are [Contradiction, Neutral, Entailment]
            probs = torch.softmax(outputs.logits, dim=1)[0]

        # Explicit mapping for facebook/bart-large-mnli
        return {
            "contradiction": probs[0].item(),
            "neutral": probs[1].item(),
            "entailment": probs[2].item()
        }

    def compute_delta_entailment(self, evidence_real, evidence_fake, answer):
        """
        Calculates: Entailment(Real) - Entailment(Fake)
        Positive Score = The answer matches Real evidence better than Fake evidence.
        """
        scores_real = self.check_entailment(evidence_real, answer)
        scores_fake = self.check_entailment(evidence_fake, answer)
        
        delta = scores_real["entailment"] - scores_fake["entailment"]
        return delta