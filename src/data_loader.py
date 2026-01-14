# src/data_loader.py
from datasets import load_dataset

class DataLoader:
    def __init__(self, dataset_name="nq_open", split="validation"):
        """
        Loads the Natural Questions (NQ-Open) dataset.
        We use 'validation' split for testing because 'train' is huge.
        """
        print(f"Loading dataset: {dataset_name} ({split})...")
        # trust_remote_code needed for some HF datasets
        self.data = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    def get_batch(self, start_index=0, limit=100):
        """
        Returns a clean list of items to process.
        """
        batch = []
        # We slice the dataset to get the requested range
        subset = self.data.select(range(start_index, start_index + limit))
        
        for i, item in enumerate(subset):
            # nq_open structure: {'question': str, 'answer': [str, str...]}
            q = item['question']
            
            # We take the first valid answer as the 'Gold Truth'
            gold_answer = item['answer'][0] 
            
            batch.append({
                "id": start_index + i,
                "question": q,
                "gold_answer": gold_answer
            })
            
        return batch

# Test block
if __name__ == "__main__":
    loader = DataLoader()
    sample = loader.get_batch(limit=3)
    print("Loaded sample:", sample)