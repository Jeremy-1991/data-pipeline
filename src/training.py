import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.utils import load_params
from sklearn.preprocessing import StandardScaler

def train_model(params: dict):
    # Chemin du dataset prétraité
    preprocess_path = params["data"]["preprocess_dataset_path"]
    model_path = params["model"]["path"]
    scaler_path = params["scaler"]["path"]
    train_path = params["data"]["train_dataset_path"]
    test_path = params["data"]["test_dataset_path"]

    # Charger le dataset
    df = pd.read_csv(preprocess_path)

    # Séparer X et y
    # La colonne cible s'appelle "Exited"
    X = df.drop(["Exited", "Surname"], axis=1, errors="ignore")
    y = df["Exited"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sauvegarder les datasets bruts
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Sauvegarder le scaler pour une utilisation future
    joblib.dump(scaler, scaler_path)
    print(f"Scaler sauvegardé dans {scaler_path}")

    # Entraîner le modèle
    clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight={0:1, 1:3})
    clf.fit(X_train_scaled, y_train)

    # Sauvegarder le modèle
    joblib.dump(clf, model_path)
    print(f"Modèle sauvegardé dans {model_path}")

if __name__ == "__main__":
    try:
        params = load_params()
        train_model(params)
    except Exception as e:
        print("Erreur :", e)