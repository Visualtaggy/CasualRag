import torch
from src.generator import CausalGenerator
from src.retriever import DenseRetriever
from src.metrics import compute_hsb
from src.perturb import Perturber

def main():
    print("=== Initializing CausalRAG Pipeline ===")
    
    # 1. Setup Models
    # We let the CausalGenerator auto-detect the best device (MPS/CUDA/CPU)
    gen = CausalGenerator(model_name="gpt2") 
    ret = DenseRetriever()
    attacker = Perturber()

    # 2. Mock Knowledge Base (In reality, load this from data/raw)
    knowledge_base = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The Eiffel Tower is located in Paris, France.",
        "Apple was founded by Steve Jobs in California in 1976."
    ]
    ret.build_index(knowledge_base)

    print("\n=== Starting Experiment ===")
    
    # 3. Define the Scenario
    query = "Where is the Eiffel Tower?"
    
    # Retrieve top-1 document
    retrieved_docs = ret.retrieve(query, k=1)
    evidence_E = retrieved_docs[0] 
    
    # Generate the Baseline Answer (y)
    # We use the Generator to answer based on the original truthful evidence
    prompt_original = f"Context: {evidence_E}\nQuestion: {query}"
    answer_y = gen.generate(prompt_original)
    
    print(f"\n[1] Query: {query}")
    print(f"[2] Original Evidence (E):  {evidence_E}")
    print(f"[3] Model Answer (y):       {answer_y}")

    # 4. The Intervention: Create Counterfactual Evidence (E')
    # We ask the perturber to swap entities (e.g., Paris -> London)
    evidence_E_prime = attacker.perturb(evidence_E, strategy="adversarial")
    
    print(f"[4] Counterfactual (E'):    {evidence_E_prime}")

    # 5. Measure Sensitivity (HSB)
    # We calculate how much the model's probability distribution changes 
    # when we force it to see the LIE (E') vs the TRUTH (E) for the SAME answer.
    
    print("\n... Calculating Logits & HSB ...")
    logits_E = gen.get_logits(evidence_E, answer_y)
    logits_E_prime = gen.get_logits(evidence_E_prime, answer_y)

    hsb_score = compute_hsb(logits_E, logits_E_prime)
    
    print("-" * 40)
    print(f"HSB Score (KL Divergence): {hsb_score:.4f}")
    print("-" * 40)
    
    # Interpretation for you (the researcher)
    if hsb_score > 0.5:
        print("RESULT: HIGH SENSITIVITY.")
        print("The model noticed the evidence changed. It was 'surprised' by the lie.")
    else:
        print("RESULT: LOW SENSITIVITY (Potential Hallucination).")
        print("The model didn't care about the evidence; it relied on its own training memory.")

if __name__ == "__main__":
    main()