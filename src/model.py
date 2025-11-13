# model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

def train_model(data_path, model_type="gb"):
    """Train and evaluate a model for dropout or readmission prediction."""
    # Load processed data
    df = pd.read_csv(data_path)
    X = df.drop("readmission", axis=1)  # Adjust target column as needed
    y = df["readmission"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Select model
    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # Save model
    joblib.dump(model, "src/model.pkl")
    print("Model saved to src/model.pkl")

if __name__ == "__main__":
    train_model("data/processed_data.csv", model_type="gb")
  
