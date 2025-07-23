import numpy as np
import scipy.sparse
import pandas as pd
import math, time, inspect
from cornac.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from cornac.models import MF, VBPR, VMF, AMR, VAECF, MostPop
from villa_mmbench.utils.utils import fitWithModalities, modelIsSelected


def gridMetric(
    model,
    val_grp,
    train_fit_set,
    train_seen,
    iid_map,
    all_iids,
    USE_GPU_FOR_HPO,
    topN=10,
):
    """
    Computes the grid search metric for hyperparameter optimization.

    Parameters
    ----------
    model : object
        The model to be evaluated.
    val_grp : dict
        A dictionary mapping user IDs to their ground truth items.
    train_fit_set : Dataset
        The training dataset used for fitting the model.
    train_seen : dict
        A dictionary mapping user IDs to sets of seen items during training.
    iid_map : dict
        A mapping of item IDs to their indices in the training dataset.
    all_iids : list
        A list of all item IDs present in the training dataset.
    USE_GPU_FOR_HPO : bool
        Flag indicating whether to use GPU for hyperparameter optimization.
    topN : int, optional
        The number of top items to consider for evaluation (default is 10).

    Returns
    -------
    float
        The average of recall and normalized discounted cumulative gain (NDCG) for the model on the validation set.
    """
    # Check if GPU is available for hyperparameter optimization
    CUPY = False
    if USE_GPU_FOR_HPO:
        try:
            import cupy as cp

            CUPY = True
            print("✔ CuPy enabled")
        except ImportError:
            print("✖ CuPy not found")
            USE_GPU_FOR_HPO = False
    # Variables
    rec, ndcg = [], []
    # Loop through each user in the validation group
    for uid, gt in val_grp.items():
        if uid not in train_fit_set.uid_map:
            continue
        uidx = train_fit_set.uid_map[uid]
        scores = model.score(uidx)
        if USE_GPU_FOR_HPO and CUPY:
            scores = cp.asarray(scores)
        seen = train_seen.get(uid, set())
        cand = [(it, scores[iid_map[it]]) for it in all_iids if it not in seen]
        cand.sort(key=lambda x: float(x[1]), reverse=True)
        top = [c[0] for c in cand[:topN]]
        rec.append(len(set(top) & set(gt)) / len(gt) if gt else 0)
        dcg = sum(1 / math.log2(r + 2) for r, it in enumerate(top) if it in gt)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(len(gt), topN)))
        ndcg.append(dcg / idcg if idcg else 0)
    return 0.5 * (np.mean(rec) + np.mean(ndcg))


def grid(dataDict, cls, name, scenario, param_grid, *fit_args):
    """
    Performs hyperparameter optimization (HPO) using grid search for a given model class.

    Parameters
    ----------
    dataDict : dict
        A dictionary containing the configuration and dataset information.
    cls : class
        The model class to be optimized.
    name : str
        The name of the model being optimized.
    scenario : str
        The scenario or context in which the model is being optimized.
    param_grid : list of dict
        A list of dictionaries containing hyperparameter configurations to be tested.
    fit_args : tuple
        Additional arguments required for fitting the model.

    Returns
    -------
    tuple
        A tuple containing the best model instance and its corresponding hyperparameters.
        The model is fitted with the training set and evaluated on the validation set.
    """
    config = dataDict["config"]
    val_grp = dataDict["val_grp"]
    iid_map = dataDict["iid_map"]
    all_iids = dataDict["all_iids"]
    train_seen = dataDict["train_seen"]
    train_fit_set = dataDict["train_fit_set"]
    # Variables
    SEED = config["experiment"]["seed"]
    VERBOSE = config["experiment"]["verbose"]
    PARALLEL_HPO = config["experiment"]["parallel_hpo"]
    USE_GPU_FOR_HPO = config["experiment"]["use_gpu_for_hpo"]
    #
    start = time.time()
    print(f"🔄 HPO {name} {scenario} - {len(param_grid)} configs")

    def _eval(p):
        p2 = p.copy()
        if USE_GPU_FOR_HPO and "use_gpu" in inspect.signature(cls).parameters:
            p2["use_gpu"] = True
        m = cls(seed=SEED, **p2)
        fitWithModalities(m, *fit_args)
        s = gridMetric(
            m, val_grp, train_fit_set, train_seen, iid_map, all_iids, USE_GPU_FOR_HPO
        )
        if VERBOSE:
            print(f"    ↳ {p2}  → {s:.4f}")
        return s, m, p2

    if PARALLEL_HPO and len(param_grid) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(param_grid))) as ex:
            results = list(ex.map(_eval, param_grid))
    else:
        results = [_eval(p) for p in param_grid]
    best = max(results, key=lambda x: x[0])
    print(
        f"✔ best {name} {scenario} = {best[2]} ({best[0]:.4f}) "
        f"[{time.time()-start:.1f}s]"
    )
    return best[1], best[2]


def gridSearch(
    config: dict, train_df: pd.DataFrame, train_set: pd.DataFrame, modalities_dict: dict
):
    """
    Performs hyperparameter optimization (HPO) using grid search for multiple models.
    This function prepares the dataset, defines the hyperparameter grid for each model,
    and runs the grid search to find the best configurations.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing experiment settings.
    train_df : pd.DataFrame
        The training DataFrame containing user-item interactions.
    train_set : Dataset
        The training dataset object containing user-item interactions.
    modalities_dict : dict
        A dictionary containing modalities for different models.

    Returns
    -------
    dict
        A dictionary containing the final models after hyperparameter tuning.
        The keys are tuples of model names and their variants, and the values are the fitted model instances.
    """
    # Variables
    models_cfg = {}
    final_models = {}
    SEED = config["experiment"]["seed"]
    VERBOSE = config["experiment"]["verbose"]
    N_EPOCHS = config["experiment"]["n_epochs"]
    MODEL_CHOICE = config["modality"]["model_choice"]
    FAST_Prtye = config["experiment"]["fast_prototype"]
    USE_GPU_FOR_HPO = config["experiment"]["use_gpu_for_hpo"]
    # Monkey‑patch so that csr_matrix.A → csr_matrix.toarray()
    if not hasattr(scipy.sparse.csr_matrix, "A"):
        scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
    #
    print(f"\nPreparing GridSearch procedure ...")
    train_fit_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
    val_grp = val_df.groupby("user_id")["item_id"].apply(list).to_dict()
    train_seen = train_fit_df.groupby("user_id")["item_id"].apply(set).to_dict()
    train_fit_set = Dataset.from_uir(
        train_fit_df[["user_id", "item_id", "rating"]].values.tolist()
    )
    all_iids, iid_map = train_fit_set.item_ids, train_fit_set.iid_map
    # Prepare the dataset with modalities
    print("✔ Preparing dataset with modalities ...")
    GR_MF = [
        {"k": k, "learning_rate": lr, "lambda_reg": 0.01, "max_iter": 50}
        for k in (32, 64, 128)
        for lr in (0.01, 0.005)
    ][0:5]
    GR_VAECF = [
        {"k": k, "learning_rate": lr, "beta": 0.01}
        for k in (32, 64, 128)
        for lr in (0.001, 0.0005)
    ][0:5]
    if FAST_Prtye:
        GR_VBPR = [
            {
                "k": k,
                "k2": k2,
                "learning_rate": lr,
                "lambda_w": 0.01,
                "lambda_b": 0.01,
                "n_epochs": 1,
            }
            for k in (32, 64, 128)
            for k2 in (8, 16)
            for lr in (0.001,)
        ][0:5]
    else:
        GR_VBPR = [
            {
                "k": k,
                "k2": k2,
                "learning_rate": lr,
                "lambda_w": 0.01,
                "lambda_b": 0.01,
                "n_epochs": N_EPOCHS,
            }
            for k in (32, 64, 128)
            for k2 in (8, 16)
            for lr in (0.001,)
        ][0:5]
    if FAST_Prtye:
        GR_VMF = [
            {"k": k, "learning_rate": lr, "n_epochs": 1}
            for k in (32, 64, 128)
            for lr in (0.01,)
        ][0:5]
    else:
        GR_VMF = [
            {"k": k, "learning_rate": lr, "n_epochs": N_EPOCHS}
            for k in (32, 64, 128)
            for lr in (0.01,)
        ][0:5]
    if FAST_Prtye:
        GR_AMR = [
            {"k": k, "k2": k2, "learning_rate": lr, "n_epochs": 1}
            for k in (32, 64, 128)
            for k2 in (16, 32)
            for lr in (0.001,)
        ][0:5]
    else:
        GR_AMR = [
            {"k": k, "k2": k2, "learning_rate": lr}
            for k in (32, 64, 128)
            for k2 in (16, 32)
            for lr in (0.001,)
        ][0:5]
    # Run grid search for each model
    print("✔ Starting HPO ...")
    dataDict = {
        "val_grp": val_grp,
        "train_seen": train_seen,
        "iid_map": iid_map,
        "all_iids": all_iids,
        "train_fit_set": train_fit_set,
        "config": config,
    }
    if modelIsSelected("MF", MODEL_CHOICE):
        models_cfg["MF"] = grid(dataDict, MF, "MF", "(na)", GR_MF, train_fit_set)
    if modelIsSelected("VAECF", MODEL_CHOICE):
        models_cfg["VAECF"] = grid(
            dataDict, VAECF, "VAECF", "(na)", GR_VAECF, train_fit_set
        )
    if modelIsSelected("VBPR", MODEL_CHOICE):
        for mod in ("visual", "audio", "text"):
            models_cfg[f"VBPR_{mod}"] = grid(
                dataDict,
                VBPR,
                "VBPR",
                mod,
                GR_VBPR,
                train_fit_set,
                modalities_dict["concat"][f"{mod}_image"],
            )
        for mv in modalities_dict:
            if mv == "concat":
                continue
            models_cfg[f"VBPR_{mv}"] = grid(
                dataDict,
                VBPR,
                "VBPR",
                mv,
                GR_VBPR,
                train_fit_set,
                modalities_dict[mv]["all_image"],
            )
    if modelIsSelected("VMF", MODEL_CHOICE):
        for mod in ("visual", "audio", "text"):
            models_cfg[f"VMF_{mod}"] = grid(
                dataDict,
                VMF,
                "VMF",
                mod,
                GR_VMF,
                train_fit_set,
                modalities_dict["concat"][f"{mod}_image"],
            )
        for mv in modalities_dict:
            if mv == "concat":
                continue
            models_cfg[f"VMF_{mv}"] = grid(
                dataDict,
                VMF,
                "VMF",
                mv,
                GR_VMF,
                train_fit_set,
                modalities_dict[mv]["all_image"],
            )
    if modelIsSelected("AMR", MODEL_CHOICE):
        for mod in ("visual", "audio", "text"):
            models_cfg[f"AMR_{mod}"] = grid(
                dataDict,
                AMR,
                "AMR",
                mod,
                GR_AMR,
                train_fit_set,
                modalities_dict["concat"][f"{mod}_image"],
                modalities_dict["concat"]["all_feature"],
            )
        for mv in modalities_dict:
            if mv == "concat":
                continue
            models_cfg[f"AMR_{mv}"] = grid(
                dataDict,
                AMR,
                "AMR",
                mv,
                GR_AMR,
                train_fit_set,
                modalities_dict[mv]["all_image"],
                modalities_dict[mv]["all_feature"],
            )
    print(f"✔ HPO done - {len(models_cfg)} configs kept")
    # Re-fit the best models with the full training set
    if modelIsSelected("TopPop", MODEL_CHOICE):
        mp = MostPop()
        mp.fit(train_set)
        final_models[("TopPop", "NA")] = mp
    for tag, (best_model, cfg) in models_cfg.items():
        mdl, variant = tag.split("_", 1) if "_" in tag else (tag, "NA")
        extras = {}
        if (
            USE_GPU_FOR_HPO
            and "use_gpu" in inspect.signature(best_model.__class__).parameters
        ):
            extras["use_gpu"] = True
        if mdl in {"MF", "VAECF"}:
            new = best_model.__class__(seed=SEED, **cfg, **extras)
            fitWithModalities(new, train_set)
            final_models[(mdl, variant)] = new
        else:
            if variant in ("visual", "audio", "text"):
                img = modalities_dict["concat"][f"{variant}_image"]
                feat = None
            else:
                img = modalities_dict[variant]["all_image"]
                feat = modalities_dict[variant].get("all_feature")
            new = best_model.__class__(seed=SEED, **cfg, **extras)
            fitWithModalities(new, train_set, img, feat)
            final_models[(mdl, variant)] = new
        if VERBOSE:
            print(f"✔ Re-fit {mdl}_{variant}")
    # Return the final models
    print(f"✔ Total final models = {len(final_models)}")
    return final_models
