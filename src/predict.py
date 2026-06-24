import joblib
import pandas as pd

def predict(data_path="data/new_customers.csv"):
    try:
        model = joblib.load('models/best_model.pkl')
    except FileNotFoundError:
        print("Model not found. Please run train.py first.")
        return
        
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("No new data found for prediction. Using dummy data.")
        df = pd.DataFrame({
            'age': [30, 45],
            'income': [50000, 80000],
            'loan_amount': [10000, 5000],
            'credit_history': [1, 1]
        })
        
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    
    df['prediction'] = preds
    df['probability'] = probs
    print(df)
    
if __name__ == "__main__":
    predict()
