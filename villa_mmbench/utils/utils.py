import os
import copy
import yaml
import numpy as np

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

def applyKcore(df, k):
    """
    Apply k-core filtering to the dataset.
    This function filters the DataFrame to retain only users and items
    that have at least k interactions.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing user-item interactions.
    k : int
        The minimum number of interactions required for users and items to be retained.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame containing only users and items with at least k interactions.
    """
    changed = True
    while changed:
        before = len(df)
        vc = df.user_id.value_counts(); df=df[df.user_id.isin(vc[vc>=k].index)]
        vc = df.item_id.value_counts(); df=df[df.item_id.isin(vc[vc>=k].index)]
        changed = len(df)<before
    return df

def parseSafe(s: str) -> np.ndarray:
    """
    Converts a string representation of a vector into a NumPy array.

    Parameters
    ----------
    s : str
        The string representation of the vector, where elements are separated by commas or spaces.

    Returns
    -------
    np.ndarray
        A NumPy array containing the elements of the vector, with non-finite values replaced by 0.0.
    """
    vec = np.fromstring(str(s).replace(',', ' '), sep=' ', dtype=np.float32)
    if not np.all(np.isfinite(vec)):
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec

def modelIsSelected(tag: str, model: str)->bool:
    """
    Checks if the given tag corresponds to a model selected in the configuration.

    Parameters
    ----------
    tag : str
        The tag representing the model.
    """
    return ((model=='cf' and tag in {'MF','VAECF','TopPop'}) or
            (model=='vbpr' and tag=='VBPR') or
            (model=='vmf' and tag=='VMF') or
            (model=='amr' and tag=='AMR'))

def fitWithModalities(model, base_ds, item_img=None, item_feat=None):
    """
    Fits the model with the given dataset and item modalities.

    Parameters
    ----------
    model : object
        The model to be fitted.
    base_ds : Dataset
        The base dataset containing user-item interactions.
    item_img : np.ndarray, optional
        The item image features to be used in the dataset.
    item_feat : np.ndarray, optional
        The item feature embeddings to be used in the dataset.
    """
    ds = copy.deepcopy(base_ds)
    if item_img  is not None: ds.item_image   = item_img
    if item_feat is not None: ds.item_feature = item_feat
    model.fit(ds)