"""
Visual Embeddings Module for ViLLA-MMBench.

This module handles the loading and processing of visual embeddings from MMTF-14K dataset.
It supports two main visual feature variants:
- CNN: Convolutional Neural Network based features (averaged)
- AVF: Audio-Visual Fusion features (averaged)

The embeddings are pre-computed and stored in CSV files for each variant.
"""

import pandas as pd
from typing import Dict, Union
from villa_mmbench.utils.utils import parseSafe

# Base URL for visual embeddings data
VIS_BASE: str = (
    "https://raw.githubusercontent.com/RecSys-lab/"
    "reproducibility_data/refs/heads/main/fused_textual_visual/"
)

# Mapping of visual feature variants to their respective file paths
VIS_MAP: Dict[str, str] = {
    "cnn": "fused_llm_mmtf_avg.csv",
    "avf": "fused_llm_mmtf_avf_avg.csv",
}


def loadVisual(config: Dict[str, Union[Dict, str, bool]]) -> pd.DataFrame:
    """
    Load and process visual embeddings based on the specified variant.

    This function loads pre-computed visual embeddings from the MMTF-14K dataset.
    It supports different variants of visual features (CNN or AVF) and handles
    the parsing of embeddings from string to numpy arrays.

    Parameters
    ----------
    config : Dict[str, Union[Dict, str, bool]]
        Configuration dictionary containing:
        - modality.visual_variant: Type of visual embeddings to use ('cnn' or 'avf')
        - experiment.verbose: Whether to print progress information

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - itemId: Unique identifier for each item
        - visual: Visual embeddings as numpy arrays

    Raises
    ------
    KeyError
        If an unsupported visual variant is specified
    ValueError
        If the embeddings file cannot be loaded or parsed
    """
    # Extract configuration parameters
    parse = parseSafe
    variant = config["modality"]["visual_variant"]
    verbose = config["experiment"]["verbose"]

    # Validate visual variant
    if variant not in VIS_MAP:
        raise KeyError(f"Unsupported visual variant: {variant}. "
                      f"Choose from: {list(VIS_MAP.keys())}")

    print(f"\nPreparing 'Visual - {variant}' data ...")

    try:
        # Read the CSV file
        df = pd.read_csv(VIS_BASE + VIS_MAP[variant])
        
        # Parse embeddings from string to numpy arrays
        df["visual"] = df.embedding.map(parse)
        
        if verbose:
            print(f"[Visual] Loaded items = {len(df):,}")
        
        return df[["itemId", "visual"]]
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty data file found for visual variant: {variant}")
    except Exception as e:
        raise ValueError(f"Failed to load visual embeddings for variant {variant}: {str(e)}")
