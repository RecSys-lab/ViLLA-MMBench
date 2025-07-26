"""
Audio Embeddings Module for ViLLA-MMBench.

This module handles the loading and processing of audio embeddings from MMTF-14K dataset and contains two sources:
- Block-level Features (BLF) with correlation, delta, log, and spectral features
- i-vector features for audio representation

The module supports two main audio variants:
- 'i_ivec': i-vector based audio embeddings
- 'blf': Bottleneck Layer Features combining multiple MMTF features with PCA
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, List
from sklearn.decomposition import PCA
from villa_mmbench.utils.utils import parseSafe
from sklearn.preprocessing import StandardScaler

# Base URL for audio embeddings data
AUD_BASE: str = (
    "https://raw.githubusercontent.com/RecSys-lab/"
    "reproducibility_data/refs/heads/main/fused_textual_audio/"
)

# Mapping of audio feature types to their respective file paths
AUD_FILE_MAP: Dict[str, str] = {
    "mmtf_corr": "fused_llm_mmtf_audio_correlation.csv",
    "mmtf_delta": "fused_llm_mmtf_audio_delta.csv",
    "mmtf_log": "fused_llm_mmtf_audio_log.csv",
    "mmtf_spect": "fused_llm_mmtf_audio_spectral.csv",
    "i_ivec": "i-vector/fused_llm_mmtf_audio_IVec_splitItem_fold_1_gmm_128_tvDim_20.csv",
}


def readAudioCsv(url: str) -> pd.DataFrame:
    """
    Read and parse audio embeddings from a CSV file.

    This function handles the preprocessing of raw audio embedding data by:
    1. Reading the CSV file
    2. Dropping unnecessary columns
    3. Standardizing column names
    4. Converting embeddings from string to numpy arrays

    Parameters
    ----------
    url : str
        The URL of the CSV file containing audio embeddings.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - itemId: Unique identifier for each item
        - embeddings: numpy array of audio embeddings

    Notes
    -----
    The function expects the CSV to have at least 'itemId' and 'embedding' columns.
    Optional columns 'title' and 'genres' will be dropped if present.
    """
    # Parse embeddings safely
    parse = parseSafe

    # Read CSV with optimized memory usage
    df = pd.read_csv(url, low_memory=False)

    # Clean up DataFrame
    df.drop(columns=["title", "genres"], errors="ignore", inplace=True)
    df.rename(columns={"embedding": "embeddings"}, inplace=True)

    # Convert embeddings from string to numpy arrays
    df["embeddings"] = (df["embeddings"]
                        .astype(str)
                        .str.replace(",", " ")
                        .apply(parse))

    return df[["itemId", "embeddings"]]


def loadAudio(config: Dict[str, Union[Dict, str, int, bool]]) -> pd.DataFrame:
    """
    Load and process audio embeddings based on the specified variant.

    This function supports two types of audio embeddings:
    1. i-vector: Direct loading of i-vector features
    2. blf (Bottleneck Layer Features): Combines multiple MMTF features with PCA

    Parameters
    ----------
    config : Dict[str, Union[Dict, str, int, bool]]
        Configuration dictionary containing:
        - experiment.seed: Random seed for reproducibility
        - experiment.verbose: Whether to print progress information
        - modality.audio_variant: Type of audio embeddings to use ('i_ivec' or 'blf')

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - itemId: Unique identifier for each item
        - audio: Processed audio embeddings as numpy arrays

    Raises
    ------
    ValueError
        If an unsupported audio variant is specified
    """
    # Extract configuration parameters
    pca_ratio: float = 0.95  # Fixed PCA ratio for dimensionality reduction
    seed: int = config["experiment"]["seed"]
    verbose: bool = config["experiment"]["verbose"]
    variant: str = config["modality"]["audio_variant"]

    print(f"\nPreparing 'Audio - {variant}' data ...")

    # Handle i-vector variant
    if variant == "i_ivec":
        df = readAudioCsv(AUD_BASE + AUD_FILE_MAP["i_ivec"])
        df.rename(columns={"embeddings": "audio"}, inplace=True)
        if verbose:
            print(f"[Audio] Loaded i-vector items = {len(df):,}")
        return df

    # Handle bottleneck layer features (blf) variant
    if variant == "blf":
        # Load all MMTF features
        dfs: List[pd.DataFrame] = []
        mmtf_keys = ["mmtf_corr", "mmtf_delta", "mmtf_log", "mmtf_spect"]

        for key in mmtf_keys:
            df = readAudioCsv(AUD_BASE + AUD_FILE_MAP[key])
            df = df.rename(columns={"embeddings": f"{key}_emb"})
            dfs.append(df)

        # Merge all features
        dfm = dfs[0]
        for d in dfs[1:]:
            dfm = dfm.merge(d, on="itemId", how="inner")

        # Concatenate all embeddings
        dfm["concat"] = dfm.apply(
            lambda r: np.concatenate([
                r["mmtf_corr_emb"],
                r["mmtf_delta_emb"],
                r["mmtf_log_emb"],
                r["mmtf_spect_emb"],
            ]),
            axis=1,
        )

        # Apply PCA for dimensionality reduction
        X = np.vstack(dfm["concat"].values)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=pca_ratio, svd_solver="full", random_state=seed)
        X_pca = pca.fit_transform(X_scaled).astype(np.float32)

        # Create final DataFrame
        df_audio = pd.DataFrame({
            "itemId": dfm["itemId"],
            "audio": list(X_pca)
        })

        if verbose:
            print(f"[Audio] PCA {pca_ratio*100}% dims = {X_pca.shape[1]}")
            print(f"[Audio] BLF items = {len(df_audio):,}")

        return df_audio

    raise ValueError(f"Unsupported audio variant: {variant}")
