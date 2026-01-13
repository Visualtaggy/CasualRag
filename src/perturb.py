import spacy
import random

# Load the NLP model once
nlp = spacy.load("en_core_web_sm")

class Perturber:
    def __init__(self):
        # Expanded dictionary of replacements
        self.replacements = {
            "GPE": ["London", "Berlin", "Tokyo", "Moscow", "Sydney", "New York"],
            "DATE": ["2001", "1999", "2025", "yesterday", "last century"],
            "PERSON": ["Alan Turing", "Elon Musk", "Ada Lovelace", "John Doe"],
            "ORG": ["Google", "OpenAI", "Umbrella Corp", "Cyberdyne Systems"],
            "FAC": ["The Empire State Building", "The Pyramids", "The White House"] # Added support for Facilities
        }

    def perturb(self, text, strategy="adversarial"):
        """
        Automatically creates a counterfactual version of the text.
        """
        doc = nlp(text)
        new_text = text
        
        # 1. Filter entities: Only keep ones we actually have replacements for
        valid_entities = [
            (ent.text, ent.label_) 
            for ent in doc.ents 
            if ent.label_ in self.replacements
        ]
        
        # Debug print to see what spaCy found
        # print(f"DEBUG: Found valid entities: {valid_entities}")
        
        if not valid_entities:
            print("WARNING: No swappable entities found in text!")
            return text 

        # 2. Adversarial: Swap a named entity with a fake one
        if strategy == "adversarial":
            # Retry loop to ensure we don't accidentally pick the exact same word
            for _ in range(5): 
                target_text, label = random.choice(valid_entities)
                
                # Get options for this label
                options = [opt for opt in self.replacements[label] if opt != target_text]
                
                if options:
                    fake_value = random.choice(options)
                    new_text = new_text.replace(target_text, fake_value)
                    
                    # If the text actually changed, we are done
                    if new_text != text:
                        return new_text
                    
        return new_text

# Quick test block
if __name__ == "__main__":
    p = Perturber()
    original = "The Eiffel Tower is located in Paris, France."
    print(f"Original: {original}")
    print(f"Attack:   {p.perturb(original)}")