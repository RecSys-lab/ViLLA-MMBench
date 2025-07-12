# !/usr/bin/env python3

from utils.utils import readConfigs

def main():
    # Read configuration file
    config = readConfigs('config.yml')
    print(f"Starting '{config['general']['name']}' ... Configurations loaded successfully!")
    print("Exiting the framework ...")


if __name__ == "__main__":
    main()