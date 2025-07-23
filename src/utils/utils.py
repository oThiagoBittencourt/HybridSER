import json
import os

# --- Função para Carregar a Configuração ---
def load_config():
    """
    Finds and loads config.json from project root.
    Returns a dict with configs
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config.json') 

    # Abre e lê o arquivo JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Erro: O arquivo de configuração não foi encontrado em '{config_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{config_path}' não é um JSON válido.")
        return None
