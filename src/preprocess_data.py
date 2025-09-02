import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import load_params

def create_preprocessor():
    numeric_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                        "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    categorical_features = ["Geography", "Gender"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop=None, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"  # Supprime "Surname" et "Exited"
    )

    return preprocessor

def preprocess_data(params: dict):
    # Récupération des chemins
    clean_path = params["data"]["clean_dataset_path"]
    preprocess_path = params["data"]["preprocess_dataset_path"]
    preprocessor_path = params["preprocessor"]["path"]

    # Charger le dataset nettoyé
    df = pd.read_csv(clean_path)

    # Séparer X et y
    # La colonne cible s'appelle "Exited"
    X = df.drop(columns=["Exited", "Surname"], errors="ignore")
    y = df["Exited"]

    # Créer et fit le préprocesseur
    preprocessor = create_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Colonnes finales
    cat_cols = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(
        ["Geography", "Gender"]
    )
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary"]
    all_cols = list(num_cols) + list(cat_cols)

    df_preprocessed = pd.DataFrame(X_transformed, columns=all_cols)
    df_preprocessed["Exited"] = y.values

    # Sauvegarder le dataset prétraité
    df_preprocessed.to_csv(preprocess_path, index=False)
    print(f"Dataset prétraité sauvegardé dans {preprocess_path}")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor sauvegardé dans {preprocessor_path}")

    return df


if __name__ == "__main__":
    params = load_params()
    preprocess_data(params)