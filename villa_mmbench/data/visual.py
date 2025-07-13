import pandas as pd
from villa_mmbench.utils.utils import parseSafe

VIS_BASE = ("https://raw.githubusercontent.com/RecSys-lab/"
            "reproducibility_data/refs/heads/main/fused_textual_visual/")
VIS_MAP = {
    "cnn": "fused_llm_mmtf_avg.csv",
    "avf": "fused_llm_mmtf_avf_avg.csv",
}

def loadVisual(config: dict):
    """
    Load visual embeddings of MMTF-14K from the specified variant.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the visual variant to load.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing item IDs and their visual embeddings.    
    """
    # Variables
    parse = parseSafe
    v = config['modality']['visual_variant']
    verbose = config['experiment']['verbose']
    print(f"\nPreparing 'Visual - {v}' data ...")
    # Read the CSV file
    df = pd.read_csv(VIS_BASE + VIS_MAP[v])
    # Map columns
    df['visual'] = df.embedding.map(parse)
    if verbose:
        print(f"[Visual] Loaded items = {len(df):,}")
    return df[['itemId','visual']]