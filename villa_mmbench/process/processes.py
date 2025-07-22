import math
import itertools
import collections
import numpy as np
import pandas as pd
from villa_mmbench.utils.utils import gini, ild, kl_div


def topN(model, uid, N, train_set, iid_map, all_iids, train_seen):
    """
    Get the top N recommendations for a user based on the model's scores.

    Parameters
    ----------
    model : object
        The recommendation model used to score items.
    uid : int
        The user ID for whom recommendations are to be generated.
    N : int
        The number of top recommendations to return.
    train_set : object
        The training set containing user-item interactions.
    iid_map : dict
        A mapping from item IDs to their indices in the model.
    all_iids : list
        A list of all item IDs available for recommendation.
    train_seen : dict
        A dictionary mapping user IDs to sets of items they have already interacted with.

    Returns
    -------
    list
        A list of the top N item IDs recommended for the user.
    """
    if uid not in train_set.uid_map:
        return []
    scores = model.score(train_set.uid_map[uid])
    cand = [
        (it, scores[iid_map[it]])
        for it in all_iids
        if it not in train_seen.get(uid, set())
    ]
    cand.sort(key=lambda x: float(x[1]), reverse=True)
    return [c[0] for c in cand[:N]]


def calculateAverageMetrics(recs):
    metric_rows = []
    for col in [c for c in recs.columns if c.startswith("rec_")]:
        mdl, scn = col.split("_", 2)[1:]
        pb = recs[f"PB_{mdl}_{scn}"].mean()
        fa = recs[f"FA_{mdl}_{scn}"].mean()
        no = recs[f"NO_{mdl}_{scn}"].mean()
        di = recs[f"DI_{mdl}_{scn}"].mean()
        cb = recs[f"CB_{mdl}_{scn}"].mean()
        rc = recs[f"RC_{mdl}_{scn}"].mean()
        nd = recs[f"ND_{mdl}_{scn}"].mean()
        cr = recs[f"CR_{mdl}_{scn}"].mean()  # Cold‑start Rate @ 10
        cv = recs[f"CV_{mdl}_{scn}"].mean()  # Catalogue Coverage @ 10
        # Append the metrics for the current model and scenario
        metric_rows.append(
            {
                "model": mdl,
                "scenario": scn,
                "Recall@10": rc,
                "NDCG@10": nd,
                "ColdRate@10": cr,
                "Coverage@10": cv,  # ▲ added columns
                "PopularityBias": pb,
                "Fairness": fa,
                "Novelty": no,
                "Diversity": di,
                "CalibrationBias": cb,
            }
        )
        return metric_rows


def generateLists(config: dict, train_df, train_set, test_df, genre_dict, final_models):
    print("Generating list of items...")
    # Variables
    rows = []
    train_pop = {}
    DATASET = config["data"]["ml_version"]
    topN_k = config["recommender"]["topN_k"]
    MODEL_CHOICE = config["modality"]["model_choice"]
    COLD_TH = config["recommender"]["cold_threshold"]
    # Prepare item ID mappings
    train_seen = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    all_iids, iid_map = train_set.item_ids, train_set.iid_map
    # Prepare popularity of items
    for _, iids_r in train_set.user_data.items():
        for ii in iids_r[0]:
            train_pop[ii] = train_pop.get(ii, 0) + 1
    max_pop = max(train_pop.values())
    cold_items = {i for i, c in train_pop.items() if c <= COLD_TH}
    coverage_dict = collections.defaultdict(set)
    # Generate per-user recommendations
    for uid, grp in test_df.groupby("user_id"):
        gt = set(grp.item_id.tolist())
        train_items = train_seen.get(uid, set())
        user_genres = list(
            itertools.chain(*(genre_dict.get(str(it), []) for it in train_items))
        )
        user_gen_dist = pd.Series(user_genres).value_counts().to_dict()

        r = {"userId": uid, "train": list(train_items), "gt": list(gt)}

        for (mdl, scn), mod in final_models.items():
            rec = topN(mod, uid, topN_k, train_set, iid_map, all_iids, train_seen)
            r[f"rec_{mdl}_{scn}"] = rec

            # ---------- Cold‑start Rate ------------------------------------------
            cold_rate = sum(it in cold_items for it in rec) / len(rec) if rec else 0
            r[f"CR_{mdl}_{scn}"] = cold_rate
            coverage_dict[(mdl, scn)].update(rec)
            # ---------------------------------------------------------------------

            # ---------- existing metrics -----------------------------------------
            pop_bias = (
                np.mean([train_pop.get(it, 0) / max_pop for it in rec]) if rec else 0
            )
            fairness = 1 - gini([train_pop.get(it, 0) for it in rec])
            novelty = (
                np.mean(
                    [
                        -math.log2(train_pop.get(it, 1) / len(train_set.user_data))
                        for it in rec
                    ]
                )
                if rec
                else 0
            )

            rec_gen = [genre_dict.get(str(it), ["(none)"]) for it in rec]
            diversity = ild(rec_gen)

            rec_gen_flat = list(itertools.chain(*rec_gen))
            rec_gen_dist = pd.Series(rec_gen_flat).value_counts().to_dict()
            calib = kl_div(user_gen_dist, rec_gen_dist)

            dcg = sum(1 / math.log2(rnk + 2) for rnk, it in enumerate(rec) if it in gt)
            idcg = sum(1 / math.log2(rnk + 2) for rnk in range(min(len(gt), 10)))
            ndcg = dcg / idcg if idcg else 0
            recall = len(set(rec) & gt) / len(gt) if gt else 0

            r.update(
                {
                    f"PB_{mdl}_{scn}": pop_bias,
                    f"FA_{mdl}_{scn}": fairness,
                    f"NO_{mdl}_{scn}": novelty,
                    f"DI_{mdl}_{scn}": diversity,
                    f"CB_{mdl}_{scn}": calib,
                    f"RC_{mdl}_{scn}": recall,
                    f"ND_{mdl}_{scn}": ndcg,
                }
            )
        rows.append(r)
    recs = pd.DataFrame(rows)
    # Broadcast coverage columns
    for (mdl, scn), items in coverage_dict.items():
        coverage = len(items) / len(all_iids)
        recs[f"CV_{mdl}_{scn}"] = coverage
    #
    fn_suffix = f"{DATASET}_{MODEL_CHOICE}"
    recs.to_csv(f"outputs/reclist_df_{fn_suffix}.csv", index=False)
    print(f"✔ reco lists saved → outputs/reclist_df_{fn_suffix}.csv")
    print(recs.head(5))
    # Calculate average metrics
    metric_rows = calculateAverageMetrics(recs)
    agg = pd.DataFrame(metric_rows)
    agg.to_csv(f"outputs/agg_metrics_{fn_suffix}.csv", index=False)
    print(f"✔ metrics saved → outputs/agg_metrics_{fn_suffix}.csv")
    pd.options.display.float_format = lambda x: f"{x:8.3f}"
    print("\n═════ FINAL METRICS ═════")
    print(agg.sort_values(["model", "scenario"]).to_string(index=False))
