# Utiliser l'image de base Python 3.11
FROM python:3.11-slim-bookworm

# Définir le répertoire de travail
WORKDIR /app

# Installe make
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY . .

# Remplace les fins de ligne CRLF vers LF (règle bug de permission)
RUN apt-get update \
    && apt-get install -y dos2unix \
    && dos2unix /app/app/run.sh

RUN chmod +x /app/app/run.sh

CMD ["bash", "-c", "./app/run.sh"]