import pandas as pd
from src.utils import load_params


def preprocess_data(params: dict):
    # Récupération des chemins
    clean_path = params["data"]["clean_dataset_path"]
    preprocess_path = params["data"]["preprocess_dataset_path"]

    # Charger le dataset nettoyé
    df = pd.read_csv(clean_path)

    # --- Prétraitement spécifique ---
    # 1. Encoder Geography en dummies
    dummies = pd.get_dummies(df["Geography"], prefix="Geography", drop_first=False).astype(int)
    df = pd.concat([df.drop("Geography", axis=1), dummies], axis=1)

    # 2. Factoriser Gender
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Sauvegarder le dataset prétraité
    df.to_csv(preprocess_path, index=False)
    print(f"Dataset prétraité sauvegardé dans {preprocess_path}")

    return df


if __name__ == "__main__":
    params = load_params()
    preprocess_data(params)
