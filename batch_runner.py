import json
import os
import torch
from tqdm import tqdm
from src.generator import CausalGenerator
from src.data_loader import DataLoader
from src.perturb import Perturber
from src.metrics import compute_hsb
from src.entailment import EntailmentGrader

def run_experiment(target_count=15000, output_file="final_thesis_results.jsonl"):
    print(f"=== 🚀 Launching Production Run: Target {target_count} Items ===")
    
    # 1. Check for existing progress (Resume Capability)
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['id'])
                except:
                    continue
    print(f"Found {len(processed_ids)} items already completed. Resuming...")

    # 2. Initialize Components
    # Note: We switch to 'train' split because 'validation' only has ~3,600 items.
    # We need the massive 87k training set to reach our 15k goal.
    loader = DataLoader(split="train") 
    gen = CausalGenerator(model_name="microsoft/Phi-3-mini-4k-instruct")
    attacker = Perturber()
    grader = EntailmentGrader()

    # 3. Get Data (Fetch more than needed to account for skipped items)
    # Fetching 25,000 to ensure we get 15,000 valid attacks
    dataset = loader.get_batch(start_index=0, limit=25000)
    
    # 4. The Loop
    success_count = len(processed_ids)
    pbar = tqdm(total=target_count, initial=success_count)
    
    # Open in APPEND mode ('a') to save progress incrementally
    with open(output_file, "a") as f:
        
        for item in dataset:
            # Stop if we hit the goal
            if success_count >= target_count:
                break
            
            # Skip if already done
            if item['id'] in processed_ids:
                continue

            try:
                q = item['question']
                # Simulated Perfect Retrieval
                evidence_E = f"The answer to the question '{q}' is {item['gold_answer']}."
                
                # A. Perturb
                evidence_E_prime = attacker.perturb(evidence_E, strategy="adversarial")
                if evidence_E == evidence_E_prime:
                    continue # Skip failed attacks

                # B. Generate
                prompt = f"Context: {evidence_E}\nQuestion: {q}"
                answer_y = gen.generate(prompt)

                # C. Metrics
                logits_E = gen.get_logits(evidence_E, answer_y)
                logits_E_prime = gen.get_logits(evidence_E_prime, answer_y)
                hsb_score = compute_hsb(logits_E, logits_E_prime)

                delta_ent = grader.compute_delta_entailment(evidence_E, evidence_E_prime, answer_y)

                # D. Save
                result = {
                    "id": item['id'],
                    "question": q,
                    "evidence_original": evidence_E,
                    "evidence_attacked": evidence_E_prime,
                    "model_answer": answer_y,
                    "hsb_score": hsb_score,
                    "delta_entailment": delta_ent
                }
                
                f.write(json.dumps(result) + "\n")
                f.flush() # Force write to disk immediately
                
                success_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Skipping Item {item['id']} due to error: {e}")
                continue

    pbar.close()
    print(f"\n✅ DONE! Collected {success_count} samples in {output_file}")

if __name__ == "__main__":
    # We DO NOT delete the file here, so we can resume if needed.
    run_experiment(target_count=15000)