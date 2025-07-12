# !/usr/bin/env python3

from utils.utils import readConfigs
from data.movielens import prepareML

def main():
    # Read configuration file
    config = readConfigs('config.yml')
    print(f"Starting '{config['general']['name']}' ... Configurations loaded successfully!")
    # Step 1: Prepare MovieLens and split into train and test sets
    prepareML(config)
    print("\nExiting the framework ...")


if __name__ == "__main__":
    main()