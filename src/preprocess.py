# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(input_path, output_path):
    """Preprocess data for student dropout or patient readmission prediction."""
    # Load data
    df = pd.read_csv(input_path)
    
    # Handle missing data
    imputer = SimpleImputer(strategy='median')
    numerical_cols = ['age', 'prior_admissions', 'comorbidity_score']
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # Feature engineering: Create new features
    df['high_risk_comorbidity'] = df['comorbidity_score'].apply(lambda x: 1 if x > 3 else 0)
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_cols = ['diagnosis']
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Combine processed data
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("data/sample_data.csv", "data/processed_data.csv")
