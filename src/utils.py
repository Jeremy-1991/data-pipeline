import yaml

def load_params(path: str = "params.yaml") -> dict:
    """
    Charge les paramètres d'un fichier YAML et les renvoie sous forme de dictionnaire.

    Args:
        path (str): Chemin du fichier params.yaml. Par défaut "params.yaml".

    Returns:
        dict: Contenu du fichier YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params