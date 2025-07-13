# !/usr/bin/env python3

from data.text import loadText
from data.audio import loadAudio
from data.visual import loadVisual
from utils.utils import readConfigs
from data.movielens import prepareML

def main():
    # Read configuration file
    config = readConfigs('config.yml')
    print(f"Starting '{config['general']['name']}' ... Configurations loaded successfully!")
    # Step 1: Prepare MovieLens and split into train and test sets
    prepareML(config)
    # Step 2: Load text, visual, and audio embeddings
    vis_df = loadVisual(config)
    aud_df = loadAudio(config)
    txt_df = loadText(config)
    print("\nExiting the framework ...")


if __name__ == "__main__":
    main()