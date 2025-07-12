import os
import yaml

# Multimodal fusion variants
MULTI_VARIANTS = [
    ('concat', None),
    ('pca',   0.95),
    ('cca',   40),
]

def readConfigs(file_path: str) -> dict:
    """
    Reads the `config.yml` file and returns its contents as a dictionary.

    Parameters
    ----------
        file_path (str): The path to the YAML file.
    
    Returns
    -------
        dict: The contents of the YAML file as a dictionary.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, '..', file_path)
    with open(full_path, 'r') as file:
        return yaml.safe_load(file)