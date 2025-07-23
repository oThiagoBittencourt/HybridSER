import json
import os

def load_config():
    """
    Finds and loads config.json from project root.
    Returns a dict with configs
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config.json') 

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config
