import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from villa_mmbench.utils.utils import parseSafe
from sklearn.preprocessing import StandardScaler

AUD_BASE = ("https://raw.githubusercontent.com/RecSys-lab/"
            "reproducibility_data/refs/heads/main/fused_textual_audio/")
AUD_FILE_MAP = {
    "mmtf_corr"  : "fused_llm_mmtf_audio_correlation.csv",
    "mmtf_delta" : "fused_llm_mmtf_audio_delta.csv",
    "mmtf_log"   : "fused_llm_mmtf_audio_log.csv",
    "mmtf_spect" : "fused_llm_mmtf_audio_spectral.csv",
    "i_ivec"  : "i-vector/fused_llm_mmtf_audio_IVec_splitItem_fold_1_gmm_128_tvDim_20.csv",
}

def readAudioCsv(url):
    """
    Read audio embeddings from a CSV file and parse the embeddings.

    Parameters
    ----------
    url : str
        The URL of the CSV file containing audio embeddings.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing item IDs and their parsed audio embeddings.
    """
    # Variables
    parse = parseSafe
    df = pd.read_csv(url, low_memory=False)
    # Process DataFrame
    df.drop(columns=['title','genres'],errors='ignore',inplace=True)
    df.rename(columns={'embedding':'embeddings'},inplace=True)
    df['embeddings']=df['embeddings'].astype(str).str.replace(',',' ')
    df['embeddings']=df['embeddings'].apply(parse)
    # Return DataFrame with itemId and embeddings
    return df[['itemId','embeddings']]

def loadAudio(config: dict):
    """
    Load audio embeddings of MMTF-14K from the specified variant.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the audio variant to load.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing item IDs and their audio embeddings.
    """
    # Variables
    pca_ratio = 0.95
    SEED = config['experiment']['seed']
    verbose = config['experiment']['verbose']
    variant = config['modality']['audio_variant']
    print(f"\nPreparing 'Audio - {variant}' data ...")
    # Check variant
    if variant=='i_ivec':
        df = readAudioCsv(AUD_BASE + AUD_FILE_MAP['i_ivec'])
        df.rename(columns={'embeddings':'audio'},inplace=True)
        if verbose: print(f"[Audio] Loaded i-vector items = {len(df):,}")
        return df
    if variant=='blf':
        dfs=[]
        for key in ('mmtf_corr','mmtf_delta','mmtf_log','mmtf_spect'):
            dfs.append(readAudioCsv(AUD_BASE+AUD_FILE_MAP[key]).rename(columns={'embeddings':f'{key}_emb'}))
        dfm = dfs[0]
        for d in dfs[1:]: dfm=dfm.merge(d,on='itemId',how='inner')
        dfm['concat'] = dfm.apply(lambda r:np.concatenate([r['mmtf_corr_emb'],r['mmtf_delta_emb'],r['mmtf_log_emb'],r['mmtf_spect_emb']]), axis=1)
        X = np.vstack(dfm['concat'].values)
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components = pca_ratio, svd_solver = 'full', random_state = SEED)
        Xp = pca.fit_transform(Xs).astype(np.float32)
        df_audio = pd.DataFrame({'itemId':dfm['itemId'],'audio':list(Xp)})
        if verbose: print(f"[Audio] BLF-concat→PCA95 dims = {Xp.shape[1]:<4} var={pca.explained_variance_ratio_.sum():.2f} items={len(df_audio):,}")
        return df_audio
    raise ValueError(f"Unknown audio variant: {variant}")