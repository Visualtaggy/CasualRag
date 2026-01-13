from src.generator import CausalGenerator
from src.retriever import DenseRetriever
from src.metrics import compute_hsb

def main():
    # 1. Setup
    # Note: Use a smaller model like 'gpt2' just for testing on CPU if needed
    gen = CausalGenerator(model_name="gpt2", device="cpu") 
    ret = DenseRetriever()

    # 2. Mock Data (In reality, load this from data/raw)
    knowledge_base = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The Eiffel Tower is located in Paris."
    ]
    ret.build_index(knowledge_base)

    # 3. The Scenario
    query = "Where is the Eiffel Tower?"
    retrieved_docs = ret.retrieve(query, k=1)
    evidence_E = retrieved_docs[0] # "The Eiffel Tower is located in Paris."
    
    # Generate Baseline Answer
    print(f"Query: {query}")
    print(f"Evidence (E): {evidence_E}")
    answer_y = gen.generate(f"Context: {evidence_E}\nQuestion: {query}")
    print(f"Model Answer (y): {answer_y}")

    # 4. The Intervention (Counterfactual)
    # We simulate a 'perturbed' evidence where Paris -> London
    evidence_E_prime = "The Eiffel Tower is located in London."
    print(f"Counterfactual Evidence (E'): {evidence_E_prime}")

    # 5. Measure Sensitivity (HSB)
    # We get logits for the SAME answer 'y' but under different contexts
    logits_E = gen.get_logits(evidence_E, answer_y)
    logits_E_prime = gen.get_logits(evidence_E_prime, answer_y)

    hsb_score = compute_hsb(logits_E, logits_E_prime)
    
    print("-" * 30)
    print(f"HSB Score (KL Divergence): {hsb_score:.4f}")
    print("Higher HSB = The model noticed the change in evidence.")
    print("Lower HSB = The model ignored the evidence (Hallucination/Stubbornness).")

if __name__ == "__main__":
    main()