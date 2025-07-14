import numpy as np
from data.text import loadText
from data.audio import loadAudio
from data.visual import loadVisual
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from villa_mmbench.utils.utils import MULTI_VARIANTS
from cornac.data import ImageModality, FeatureModality, Dataset

def prepareModalities(config: dict):
    # Variables
    SEED = config['experiment']['seed']
    VERBOSE = config['experiment']['verbose']
    # Fetch modalities from configuration
    vis_df = loadVisual(config) 
    aud_df = loadAudio(config)
    txt_df = loadText(config)
    # Preprocess dataframes
    for df in (vis_df, aud_df, txt_df):
        df['itemId'] = df.itemId.astype(str)
    # Merge dataframes on itemId
    common = set(vis_df.itemId) & set(aud_df.itemId) & set(txt_df.itemId)
    vis_df, aud_df, txt_df = [df[df.itemId.isin(common)].reset_index(drop=True)
                            for df in (vis_df, aud_df, txt_df)]
    merged = vis_df.merge(aud_df, on='itemId').merge(txt_df, on='itemId')
    # Guard against NaN/Inf
    for col in ('audio','visual','text'):
        merged[col] = merged[col].apply(lambda v: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    # Prepare all modalities combined
    merged['all'] = merged.apply(lambda r: np.hstack([r.audio,r.visual,r.text]), axis=1)
    keep = set(merged.itemId)
    # Load train and test sets
    train_df = train_df[train_df.item_id.astype(str).isin(keep)].reset_index(drop=True)
    test_df  = test_df [test_df .item_id.astype(str).isin(keep)].reset_index(drop=True)
    train_set = Dataset.from_uir(train_df[['user_id','item_id','rating']].values.tolist())
    if VERBOSE:
        print(f"✔ Embeddings intersect items = {len(keep):,}")
    # Create modalities dictionary
    modalities_dict = {}
    def _im(col): return ImageModality(features=np.vstack(merged[col]), ids=merged.itemId, normalized=True)
    def _ft(col): return FeatureModality(features=np.vstack(merged[col]), ids=merged.itemId, normalized=True)
    modalities_dict['concat'] = {
        'audio_image': _im('audio'),
        'visual_image':_im('visual'),
        'text_image':  _im('text'),
        'all_image':   _im('all'),
        'all_feature': _ft('all'),
    }
    if VERBOSE: print("✔ Concat ready!")
    # Add modalities to the dataset
    for tag, param in MULTI_VARIANTS:
        if tag == 'concat': continue
        if tag == 'pca':
            ratio = param; name=f"pca_{int(ratio * 100)}"
            mat = StandardScaler().fit_transform(np.vstack(merged['all']))
            mat = PCA(ratio,random_state=SEED).fit_transform(mat)
            merged[name] = list(mat.astype(np.float32))
            modalities_dict[name] = {'all_image':_im(name),'all_feature':_ft(name)}
            if VERBOSE: print(f"✔ PCA {int(ratio*100)} dims={mat.shape[1]}")
        elif tag == 'cca':
            comps = param; name=f"cca_{comps}"
            half = len(merged['all'][0])//2
            big = np.vstack(merged['all']);X,Y=big[:,:half],big[:,half:]
            cca = CCA(n_components=comps).fit(X,Y)
            merged[name] = list(cca.transform(X,Y)[0].astype(np.float32))
            modalities_dict[name] = {'all_image':_im(name),'all_feature':_ft(name)}
            if VERBOSE: print(f"✔ CCA {comps} dims={comps}")
    # Return
    return train_set, test_df, modalities_dict