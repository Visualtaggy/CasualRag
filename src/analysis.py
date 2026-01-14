import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def analyze_results(file_path="results.jsonl"):
    data = []
    # robust read that handles potential empty lines
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print("No data found in results.jsonl!")
        return

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records.")
    
    # --- FIX: Calculate the missing column manually ---
    # We define 'sensitive' as HSB > 0.5 (Model was surprised)
    df['sensitive_hsb'] = df['hsb_score'] > 0.5
    
    # 1. Basic Stats
    print("\n=== Statistics ===")
    print(df[['hsb_score', 'delta_entailment']].describe())

    # 2. Correlation Plot (HSB vs Entailment)
    plt.figure(figsize=(10, 6))
    
    # Check if we have enough variation to plot
    if len(df) > 1:
        sns.scatterplot(
            data=df, 
            x='hsb_score', 
            y='delta_entailment', 
            hue='sensitive_hsb', 
            palette='viridis'
        )
        plt.title("Hallucination Sensitivity: Probability Shift vs Logical Shift")
        plt.xlabel("HSB Score (Model Surprise)")
        plt.ylabel("Delta Entailment (Logical Consistency Drop)")
        
        # Add a threshold line
        plt.axvline(0.5, color='r', linestyle='--', label="Sensitivity Threshold")
        
        plt.legend(title="Detected?")
        plt.grid(True, alpha=0.3)
        plt.savefig("sensitivity_plot.png")
        print("\nSaved plot to 'sensitivity_plot.png'")
    else:
        print("\nNot enough data points to plot yet.")

    # 3. Success Rate
    if 'sensitive_hsb' in df.columns:
        rate = df['sensitive_hsb'].mean()
        print(f"\nSensitivity Rate: {rate:.2%} of attacks were detected by the model.")

if __name__ == "__main__":
    analyze_results()