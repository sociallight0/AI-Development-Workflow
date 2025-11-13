# utils.py
import pandas as pd
import numpy as np
import argparse

def generate_synthetic_data(output_path, n_samples=1000):
    """Generate synthetic data for testing."""
    np.random.seed(42)
    data = {
        "age": np.random.randint(18, 90, n_samples),
        "prior_admissions": np.random.randint(0, 5, n_samples),
        "comorbidity_score": np.random.uniform(0, 5, n_samples),
        "diagnosis": np.random.choice(["A", "B", "C"], n_samples),
        "readmission": np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    args = parser.parse_args()
    
    if args.generate_data:
        generate_synthetic_data("data/sample_data.csv")
