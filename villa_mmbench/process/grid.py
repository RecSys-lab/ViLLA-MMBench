import numpy as np
import scipy.sparse
import pandas as pd
import math, time, inspect
from cornac.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from cornac.models import MF, VBPR, VMF, AMR, VAECF, MostPop
from villa_mmbench.utils.utils import fitWithModalities, modelIsSelected

def gridMetric(model, val_grp, train_fit_set, train_seen, iid_map, all_iids, USE_GPU_FOR_HPO, topN = 10):
    # Check if GPU is available for hyperparameter optimization
    CUPY = False
    if USE_GPU_FOR_HPO:
        try:
            import cupy as cp; CUPY = True; print("✔ CuPy enabled")
        except ImportError:
            print("✖ CuPy not found"); USE_GPU_FOR_HPO = False
    # Variables
    rec, ndcg = [], []
    # Loop through each user in the validation group
    for uid, gt in val_grp.items():
        if uid not in train_fit_set.uid_map:continue
        uidx = train_fit_set.uid_map[uid]
        scores = model.score(uidx)
        if USE_GPU_FOR_HPO and CUPY:scores = cp.asarray(scores)
        seen = train_seen.get(uid,set())
        cand = [(it, scores[iid_map[it]]) for it in all_iids if it not in seen]
        cand.sort(key=lambda x:float(x[1]), reverse = True)
        top = [c[0] for c in cand[:topN]]
        rec.append(len(set(top)&set(gt)) / len(gt) if gt else 0)
        dcg = sum(1/math.log2(r+2) for r,it in enumerate(top) if it in gt)
        idcg = sum(1/math.log2(r+2) for r in range(min(len(gt),topN)))
        ndcg.append(dcg/idcg if idcg else 0)
    return 0.5 * (np.mean(rec)+np.mean(ndcg))

def grid(cls, name, scenario, param_grid, val_grp, 
         train_fit_set, train_seen, iid_map, all_iids, config: dict, *fit_args):
    # Variables
    SEED = config['experiment']['seed']
    VERBOSE = config['experiment']['verbose']
    PARALLEL_HPO = config['experiment']['parallel_hpo']
    USE_GPU_FOR_HPO = config['experiment']['use_gpu_for_hpo']
    # 
    start = time.time();print(f"🔄 HPO {name} {scenario} - {len(param_grid)} configs")
    def _eval(p):
        p2 = p.copy()
        if USE_GPU_FOR_HPO and 'use_gpu' in inspect.signature(cls).parameters:
            p2['use_gpu']=True
        m = cls(seed=SEED,**p2)
        fitWithModalities(m,*fit_args)
        s = gridMetric(m, val_grp, train_fit_set, train_seen, iid_map, all_iids, USE_GPU_FOR_HPO)
        if VERBOSE: print(f"    ↳ {p2}  → {s:.4f}")
        return s,m,p2
    if PARALLEL_HPO and len(param_grid)>1:
        with ThreadPoolExecutor(max_workers=min(8,len(param_grid))) as ex:
            results=list(ex.map(_eval,param_grid))
    else: results=[_eval(p) for p in param_grid]
    best = max(results,key=lambda x:x[0])
    print(f"✔ best {name} {scenario} = {best[2]} ({best[0]:.4f}) "
          f"[{time.time()-start:.1f}s]")
    return best[1],best[2]

def gridSearch(config: dict, train_df: pd.DataFrame, modalities_dict: dict):
    # Variables
    models_cfg = {}
    SEED = config['experiment']['seed']
    VERBOSE = config['experiment']['verbose']
    N_EPOCHS = config['experiment']['n_epochs']
    FAST_Prtye = config['experiment']['fast_prototype']
    USE_GPU_FOR_HPO = config['experiment']['use_gpu_for_hpo']
    # Monkey‑patch so that csr_matrix.A → csr_matrix.toarray()
    if not hasattr(scipy.sparse.csr_matrix, 'A'):
        scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
    # 
    train_fit_df, val_df = train_test_split(train_df, test_size = 0.1, random_state = SEED)
    val_grp = val_df.groupby('user_id')['item_id'].apply(list).to_dict()
    train_seen = train_fit_df.groupby('user_id')['item_id'].apply(set).to_dict()
    train_fit_set = Dataset.from_uir(train_fit_df[['user_id','item_id','rating']].values.tolist())
    all_iids, iid_map = train_fit_set.item_ids, train_fit_set.iid_map
    # Prepare the dataset with modalities
    GR_MF = [{'k':k,'learning_rate':lr,'lambda_reg':0.01,'max_iter':50}
         for k in (32,64,128) for lr in (0.01,0.005)][0:5]
    GR_VAECF = [{'k':k,'learning_rate':lr,'beta':0.01}
          for k in (32,64,128) for lr in (0.001,0.0005)][0:5]
    if FAST_Prtye:
        GR_VBPR = [
            {'k': k, 'k2': k2, 'learning_rate': lr, 'lambda_w': 0.01, 'lambda_b': 0.01, 'n_epochs': 1}
            for k in (32, 64, 128)
            for k2 in (8, 16)
            for lr in (0.001,)
        ][0:5]
    else:
        GR_VBPR = [
            {'k': k, 'k2': k2, 'learning_rate': lr, 'lambda_w': 0.01, 'lambda_b': 0.01, 'n_epochs': N_EPOCHS}
            for k in (32, 64, 128)
            for k2 in (8, 16)
            for lr in (0.001,)
        ][0:5]
    if FAST_Prtye:
        GR_VMF=[{'k':k,'learning_rate':lr,'n_epochs':1}
                for k in (32,64,128) for lr in (0.01,)][0:5]
    else:
        GR_VMF=[{'k':k,'learning_rate':lr, 'n_epochs': N_EPOCHS}
                for k in (32,64,128) for lr in (0.01,)][0:5]
    if FAST_Prtye:
        GR_AMR=[{'k':k,'k2':k2,'learning_rate':lr,'n_epochs':1}
                for k in (32,64,128) for k2 in (16,32) for lr in (0.001,)][0:5]
    else:
        GR_AMR=[{'k':k,'k2':k2,'learning_rate':lr}
                for k in (32,64,128) for k2 in (16,32) for lr in (0.001,)][0:5]
    # Run grid search for each model
    if modelIsSelected('MF'):
        models_cfg['MF'] = grid(MF, 'MF', '(na)', GR_MF, val_grp, 
                                      train_fit_set, train_seen, iid_map, all_iids, config)
    if modelIsSelected('VAECF'):
        models_cfg['VAECF'] = grid(VAECF, 'VAECF', '(na)', GR_VAECF, val_grp, train_fit_set,
                                         train_seen, iid_map, all_iids, config)
    if modelIsSelected('VBPR'):
        for mod in ('visual','audio','text'):
            models_cfg[f'VBPR_{mod}'] = grid(VBPR,'VBPR',mod,GR_VBPR, val_grp, 
                                            train_fit_set, train_seen, iid_map, all_iids, config,
                                            modalities_dict['concat'][f'{mod}_image'])
        for mv in modalities_dict:
            if mv=='concat':continue
            models_cfg[f'VBPR_{mv}'] = grid(VBPR,'VBPR',mv,GR_VBPR, val_grp, 
                                        train_fit_set, train_seen, iid_map, all_iids, config,
                                        modalities_dict[mv]['all_image'])
    if modelIsSelected('VMF'):
        for mod in ('visual','audio','text'):
            models_cfg[f'VMF_{mod}'] = grid(VMF,'VMF',mod,GR_VMF, val_grp, 
                                        train_fit_set, train_seen, iid_map, all_iids, config,
                                        modalities_dict['concat'][f'{mod}_image'])
        for mv in modalities_dict:
            if mv=='concat':continue
            models_cfg[f'VMF_{mv}'] = grid(VMF,'VMF',mv,GR_VMF, val_grp, 
                                        train_fit_set, train_seen, iid_map, all_iids, config,
                                        modalities_dict[mv]['all_image'])
    if modelIsSelected('AMR'):
        for mod in ('visual','audio','text'):
            models_cfg[f'AMR_{mod}'] = grid(AMR,'AMR',mod,GR_AMR, val_grp, 
                                        train_fit_set, train_seen, iid_map, all_iids, config,
                                        modalities_dict['concat'][f'{mod}_image'],
                                        modalities_dict['concat']['all_feature'])
        for mv in modalities_dict:
            if mv=='concat':continue
            models_cfg[f'AMR_{mv}'] = grid(AMR,'AMR',mv,GR_AMR, val_grp, 
                                        train_fit_set, train_seen, iid_map, all_iids, config,
                                        modalities_dict[mv]['all_image'],
                                        modalities_dict[mv]['all_feature'])
    # Finished
    print(f"✔ HPO done - {len(models_cfg)} configs kept")