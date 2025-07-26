"""
Text Embeddings Module for ViLLA-MMBench.

This module handles the loading and processing of text embeddings from different sources:
- Original text embeddings from movie descriptions
- Augmented text embeddings from enriched descriptions

The module supports two types of text sources:
- Original raw text embeddings (unaugmented)
- Enriched description embeddings (augmented)
"""

import pandas as pd
from typing import Dict, List, Union
from villa_mmbench.utils.utils import parseSafe

# Base URLs for text embeddings data
TXT_BASE_ORIG: str = (
    "https://raw.githubusercontent.com/yasdel/Poison-RAG-Plus/"
    "main/AttackData/Embeddings_from_Augmentation_Attack_Data/"
    "ml-latest-small/"
)
TXT_BASE_AUG: str = TXT_BASE_ORIG


def loadText(config: Dict[str, Union[Dict, str, int, bool]]) -> pd.DataFrame:
    """
    Load and process text embeddings based on configuration settings.

    This function loads text embeddings from multiple CSV files, which can be either
    original or augmented embeddings. The embeddings are combined and deduplicated
    before being returned.

    Parameters
    ----------
    config : Dict[str, Union[Dict, str, int, bool]]
        Configuration dictionary containing:
        - experiment.verbose: Whether to print progress information
        - modality.llm_prefix: Prefix for text embeddings (e.g., 'llm')
        - modality.text_max_parts: Maximum number of parts to load
        - modality.text_augmented: Whether to use augmented text embeddings

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - itemId: Unique identifier for each item
        - text: Text embeddings as numpy arrays

    Notes
    -----
    The function attempts to load parts sequentially until either:
    1. The maximum number of parts is reached
    2. A part file is not found (indicating end of available parts)
    
    Duplicate itemIds are removed, keeping the first occurrence only.
    """
    # Extract configuration parameters
    dfs: List[pd.DataFrame] = []
    parse = parseSafe
    verbose: bool = config["experiment"]["verbose"]
    llm_prefix: str = config["modality"]["llm_prefix"]
    max_parts: int = config["modality"]["text_max_parts"]
    augmented: bool = config["modality"]["text_augmented"]
    
    # Determine base URL and prefix based on configuration
    base: str = TXT_BASE_AUG if augmented else TXT_BASE_ORIG
    TXT_PREFIX_ORIG: str = f"{llm_prefix}_originalraw_combined_all_part"
    TXT_PREFIX_AUGMENTED: str = f"{llm_prefix}_enriched_description_part"
    prefix: str = TXT_PREFIX_AUGMENTED if augmented else TXT_PREFIX_ORIG

    print(f"\nPreparing 'Textual - {llm_prefix}' data ...")

    # Load and process parts sequentially
    for i in range(1, max_parts + 1):
        url: str = f"{base}{prefix}{i}.csv.gz"
        try:
            # Read and process each part
            df = pd.read_csv(url, compression="gzip")
            df["text"] = df.embeddings.map(parse)
            dfs.append(df[["itemId", "text"]])
        except pd.errors.EmptyDataError:
            print(f"Warning: Empty data file found at part {i}")
            break
        except Exception as e:
            # Stop loading if any part is not found or other error occurs
            if not dfs:  # No successful loads yet
                raise RuntimeError(f"Failed to load any text embedding parts: {str(e)}")
            break

    # Ensure at least one part was loaded successfully
    if not dfs:
        raise ValueError("No text embedding parts were loaded successfully")

    # Combine all parts and remove duplicates
    out = pd.concat(dfs).drop_duplicates("itemId")
    
    if verbose:
        tag = "AUG" if augmented else "ORIG"
        print(f"[Text] {tag} parts={len(dfs)} items={len(out):,}")
    
    return out
