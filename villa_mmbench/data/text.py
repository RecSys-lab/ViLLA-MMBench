import pandas as pd
from villa_mmbench.utils.utils import parseSafe

TXT_BASE_ORIG = ("https://raw.githubusercontent.com/yasdel/Poison-RAG-Plus/"
                 "main/AttackData/Embeddings_from_Augmentation_Attack_Data/"
                 "ml-latest-small/")
TXT_BASE_AUG = TXT_BASE_ORIG

def loadText(config: dict):
    """
    Load text embeddings from the specified parts of the dataset.

    Parameters
    ----------
    max_parts : int
        The maximum number of parts to load.
    augmented : bool
        If True, load augmented text embeddings; otherwise, load original text embeddings.
    llm_prefix : str
        The prefix for the text embeddings, typically 'llm' or similar.
    verbose : bool, optional
        If True, print loading messages. Default is True.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing item IDs and their text embeddings.
    """
    # Variables
    dfs = []
    parse = parseSafe
    verbose = config['experiment']['verbose']
    llm_prefix = config['modality']['llm_prefix']
    max_parts = config['modality']['text_max_parts']
    augmented = config['modality']['text_augmented']
    base = TXT_BASE_AUG if augmented else TXT_BASE_ORIG
    TXT_PREFIX_ORIG = f"{llm_prefix}_originalraw_combined_all_part"
    TXT_PREFIX_AUGMENTED = f"{llm_prefix}_enriched_description_part"
    # Check prefix
    print(f"\nPreparing 'Textual - {llm_prefix}' data ...")
    prefix = TXT_PREFIX_AUGMENTED if augmented else TXT_PREFIX_ORIG
    # 
    for i in range(1, max_parts+1):
        url = f"{base}{prefix}{i}.csv.gz"
        try:
            df = pd.read_csv(url, compression='gzip')
            df['text'] = df.embeddings.map(parse)
            dfs.append(df[['itemId','text']])
        except:
            break
    out = pd.concat(dfs).drop_duplicates('itemId')
    if verbose:
        tag = 'AUG' if augmented else 'ORIG'
        print(f"[Text] {tag} parts={len(dfs)} items={len(out):,}")
    return out