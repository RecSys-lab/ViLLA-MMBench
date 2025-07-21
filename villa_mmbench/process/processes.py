import collections


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


def generateLists(config: dict, train_df, train_set):
    print("Generating list of items...")
    # Variables
    rows = []
    train_pop = {}
    SEED = config["experiment"]["seed"]
    topN_k = config["recommender"]["topN_k"]
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
