import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_data, preprocess_data
from evaluation import evaluate_model

def train_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier()
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob, name)
        
        if metrics['ROC-AUC'] > best_score:
            best_score = metrics['ROC-AUC']
            best_model = model
            best_name = name
            
    print(f"Best model: {best_name} with ROC-AUC {best_score:.4f}")
    joblib.dump(best_model, 'models/best_model.pkl')

if __name__ == "__main__":
    train_models()
