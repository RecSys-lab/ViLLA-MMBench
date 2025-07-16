# !/usr/bin/env python3


from utils.utils import readConfigs
from data.movielens import prepareML
from data.modalities import prepareModalities

def main():
    # Read configuration file
    config = readConfigs('config.yml')
    print(f"Starting '{config['general']['name']}' ... Configurations loaded successfully!")
    # Step 1: Prepare MovieLens and split into train and test sets
    train_df, test_df, genre_dict = prepareML(config)
    # Step 2: Load text, visual, and audio embeddings
    prepareModalities(config, train_df, test_df)
    print("\nExiting the framework ...")


if __name__ == "__main__":
    main()