import os
import zipfile
import requests
import numpy as np
import pandas as pd
from cornac.data import Dataset
from villa_mmbench.utils.utils import applyKcore

ML100K_URL  = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
ML100K_ITEM = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
ML1M_URL    = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def downloadMovieLens(url, dest, VERBOSE=True):
    """
    Download a MovieLens dataset file if it does not exist.

    Parameters
    ----------
    url : str
        The URL to download the dataset from.
    dest : str
        The destination file path where the dataset will be saved.
    VERBOSE : bool, optional
        If True, print download messages. Default is True.
    """
    if not os.path.exists(dest):
        # Go to a 'data' directory if it exists, otherwise create it
        if VERBOSE: print(f"⏬ Download {dest}")
        open(dest, 'wb').write(requests.get(url).content)
    else:
        if VERBOSE: print(f"✅ '{dest}' already exists, skipping download.")

def loadGenres(download_path_prefix: str, DATASET: str) -> pd.DataFrame:
    """
    Load genres from the MovieLens dataset.

    Parameters
    ----------
    download_path_prefix : str
        The prefix path where the dataset files are downloaded.
    DATASET : str
        The version of the MovieLens dataset ('100k' or '1m').
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing item IDs and their associated genres.
    """
    if DATASET == '100k':
        # Variables
        genre_cols = [
            "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
            "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
            "Romance","Sci-Fi","Thriller","War","Western"
        ]
        cols = ["item_id","title","release_date","video_release_date","IMDb_URL"] + genre_cols
        dest_data = os.path.join(download_path_prefix, 'u.item')
        # Read the movies file
        movies = pd.read_csv(dest_data, sep='|', header=None,
                             names=cols, encoding='latin-1')
        movies['genres'] = movies[genre_cols].apply(
            lambda row: [g for g in genre_cols if row[g]==1], axis=1)
        movies['item_id'] = movies['item_id'].astype(str)
    else:
        # Variables
        dest_folder = os.path.join(download_path_prefix, 'ml-1m')
        path = os.path.join(dest_folder, 'ml-1m/movies.dat')
        if not os.path.exists(path):
            path = os.path.join(dest_folder, 'ml-1m/ml-1m/movies.dat')
        # Read the movies file
        movies = pd.read_csv(path, sep='::', engine='python',
                             names=['item_id','title','genres'], encoding='latin-1')
        movies['item_id'] = movies['item_id'].astype(str)
        movies['genres'] = movies['genres'].map(
            lambda s: s.split('|') if isinstance(s,str) else [])
    return movies[['item_id','genres']]

def prepareML(config: dict):
    # Variables
    K_CORE = config['data']['k_core']
    SEED = config['experiment']['seed']
    DATASET = config['data']['ml_version']
    VERBOSE = config['experiment']['verbose']
    SPLIT_MODE = config['data']['split']['mode']
    TEST_RATIO = config['data']['split']['test_ratio']
    download_path_prefix = os.path.join('villa_mmbench', 'data', 'downloaded')
    # Download the dataset
    print(f"\nPreparing 'MovieLens {DATASET}' data ...")
    if DATASET == '100k':
        # Variables
        dest_data = os.path.join(download_path_prefix, 'u.data')
        dest_item = os.path.join(download_path_prefix, 'u.item')
        # Download
        downloadMovieLens(ML100K_URL, dest_data, VERBOSE)
        downloadMovieLens(ML100K_ITEM, dest_item, VERBOSE)
        # Separate ratings and items
        ratings_file, delim, eng = dest_data, '\t', None
    else:
        # Variables
        dest = os.path.join(download_path_prefix, 'ml-1m.zip')
        dest_folder = os.path.join(download_path_prefix, 'ml-1m')
        # Download
        downloadMovieLens(ML1M_URL, dest, VERBOSE)
        if not os.path.exists(dest_folder):
            print(f"⏬ Extracting '{dest}' to '{dest_folder}' ...")
            zipfile.ZipFile(dest).extractall(dest_folder)
        ratings_file = (f'{dest_folder}/ml-1m/ratings.dat'
                        if os.path.exists(f'{dest_folder}/ml-1m/ratings.dat')
                        else f'{dest_folder}/ratings.dat')
        delim, eng = '::', 'python'
    # Read ratings file
    ratings = pd.read_csv(ratings_file, sep=delim,
                      names=['user_id','item_id','rating','timestamp'],
                      engine=eng, header=None)
    if VERBOSE: print(f"✔ Ratings rows = {len(ratings):,}")
    # Load genres
    genres_df  = loadGenres(download_path_prefix, DATASET)
    genre_dict = dict(zip(genres_df.item_id, genres_df.genres))
    if VERBOSE: print(f"✔ genres loaded items = {len(genres_df):,}")
    # Apply k-core filtering (if specified)
    if K_CORE > 0:
        ratings = applyKcore(ratings, K_CORE)
        if VERBOSE: print(f"✔ After {K_CORE}-core rows = {len(ratings):,}")
    # Split the dataset into train and test sets
    np.random.seed(SEED)
    if SPLIT_MODE == 'random':
        ratings = ratings.sample(frac=1,random_state = SEED).reset_index(drop=True)
        sz = int(len(ratings) * TEST_RATIO)
        train_df, test_df = ratings.iloc[:-sz].copy(), ratings.iloc[-sz:].copy()
    elif SPLIT_MODE=='temporal':
        ratings = ratings.sort_values('timestamp')
        sz = int(len(ratings)*TEST_RATIO)
        train_df, test_df = ratings.iloc[:-sz].copy(), ratings.iloc[-sz:].copy()
    else:
        trs, tes = [], []
        for uid, grp in ratings.groupby('user_id'):
            grp = grp.sort_values('timestamp')
            tes.append(grp.iloc[-1]); trs.extend(grp.iloc[:-1].to_dict('records'))
        train_df, test_df=pd.DataFrame(trs),pd.DataFrame(tes)
    # Make the train set
    if VERBOSE: print(f"✔ Split train = {len(train_df):,}  test = {len(test_df):,}")
    # train_set = Dataset.from_uir(train_df[['user_id','item_id','rating']].values.tolist())
    # Return
    return train_df, test_df, genre_dict