import pandas as pd
from src.utils import load_params


def clean_data(params: dict):
    # Récupération des chemins
    raw_path = params["data"]["raw_dataset_path"]
    clean_path = params["data"]["clean_dataset_path"]
    columns_to_drop = params["data"]["columns_to_drop"]

    # Chargement du dataset brut
    df = pd.read_csv(raw_path)

    # Suppression des colonnes
    df.drop(columns=columns_to_drop, inplace=True)

    # Sauvegarde du dataset nettoyé
    df.to_csv(clean_path, index=False)
    print(f"Dataset nettoyé sauvegardé dans {clean_path}")

    return df


if __name__ == "__main__":
    params = load_params()
    clean_data(params)