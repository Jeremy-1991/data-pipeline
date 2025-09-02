import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.utils import load_params

def evaluate_model(params: dict):
    # Chemin du dataset prétraité
    model_path = params["model"]["path"]
    test_path = params["data"]["test_dataset_path"]

    # Charger le modèle
    model = joblib.load(model_path)

    # Charger le dataset de test
    df_test = pd.read_csv(test_path)

    # Séparer X et y
    X_test = df_test.drop(["Exited"], axis=1, errors="ignore")
    y_test = df_test["Exited"]

    # Prédiction
    y_pred = model.predict(X_test)

    # Évaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy : {acc:.4f}")

    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix :")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    params = load_params()
    evaluate_model(params)
