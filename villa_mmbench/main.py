# !/usr/bin/env python3


from utils.utils import readConfigs
from process.grid import gridSearch
from data.movielens import prepareML
from process.processes import generateLists
from data.modalities import prepareModalities


def main():
    # Read configuration file
    config = readConfigs("config.yml")
    print(
        f"Starting '{config['general']['name']}' ... Configurations loaded successfully!"
    )
    # Step 1: Prepare MovieLens and split into train and test sets
    train_df, test_df, genre_dict = prepareML(config)
    # Step 2: Load text, visual, and audio embeddings
    train_df, test_df, train_set, modalities_dict = prepareModalities(
        config, train_df, test_df
    )
    # Step 3: Hyperparameter tuning using grid search
    final_models = gridSearch(config, train_df, train_set, modalities_dict)
    # Step 4: Generate recommendations for the test set
    generateLists(config, train_df, train_set, test_df, genre_dict, final_models)
    # Step 5: That's it! The framework has completed its execution.
    print("\nExiting the framework ...")


if __name__ == "__main__":
    main()
