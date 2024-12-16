import os
import yaml
import pickle
import sys
import matplotlib.pyplot as plt

def extract_numeric_data(data, parent_key=""):
    """
    Extrait les valeurs traçables d'un dictionnaire ou d'une liste.
    """
    extracted_data = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (int, float)):
                extracted_data[full_key] = value
            elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
                extracted_data[full_key] = value
            elif isinstance(value, dict):
                extracted_data.update(extract_numeric_data(value, full_key))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            full_key = f"{parent_key}[{idx}]"
            if isinstance(value, (int, float)):
                extracted_data[full_key] = value
            elif isinstance(value, dict):
                extracted_data.update(extract_numeric_data(value, full_key))
    return extracted_data

def process_experiment(folder_path):
    try:
        # Charger config.yml
        config_path = os.path.join(folder_path, "config.yml")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Charger metrics.yaml
        metrics_path = os.path.join(folder_path, "metrics.yml")
        with open(metrics_path, 'r') as file:
            metrics = yaml.safe_load(file)

        # Charger final_dict.pickle
        final_dict_path = os.path.join(folder_path, "final_dict.pickle")
        with open(final_dict_path, 'rb') as file:
            final_dict = pickle.load(file)

        # Récupérer la précision
        accuracy = metrics.get("accuracy", "Non disponible")

        # Afficher la précision
        print(f"Accuracy: {accuracy}")

        # Extraire les données traçables de final_dict
        numeric_data = extract_numeric_data(final_dict)

        if numeric_data:
            print("Données traçables trouvées dans final_dict:")
            for key, value in numeric_data.items():
                print(f"{key}: {value}")

            # Vérifier si les données sont des listes et les afficher avant de tracer
            for key, value in numeric_data.items():
                if isinstance(value, list):
                    print(f"Valeurs extraites pour {key}: {value}")
                    plt.plot(value, label=key)

            # Tracer les données si elles sont sous forme de liste
            if len(numeric_data) > 0:
                plt.title("Graphes des données traçables")
                plt.legend()
                plt.grid()
                plt.show()
            else:
                print("Aucune donnée traçable sous forme de liste trouvée.")

        else:
            print("Aucune donnée traçable trouvée dans final_dict.")

    except Exception as e:
        print(f"Erreur lors du traitement du dossier {folder_path}: {e}")

if __name__ == "__main__":
    # Vérifier les arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_experiment_folder>")
        sys.exit(1)

    # Récupérer le chemin du dossier
    folder_path = sys.argv[1]

    # Vérifier si le dossier existe
    if not os.path.isdir(folder_path):
        print(f"Le chemin fourni n'est pas un dossier valide : {folder_path}")
        sys.exit(1)

    # Traiter le dossier
    process_experiment(folder_path)
