import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data dir creator")
    parser.add_argument("-r", "--root", action="store", default="../../", help="Root dir to place data dir")

    args = parser.parse_args()

    root_dir = args.root

    if not os.path.exists(os.path.join(root_dir, "data")):
            os.mkdir(os.path.join(root_dir, "data"))

    if not os.path.exists(os.path.join(root_dir, "data", "raw")):
            os.mkdir(os.path.join(root_dir, "data", "raw"))
    
    if not os.path.exists(os.path.join(root_dir, "data", "processed")):
            os.mkdir(os.path.join(root_dir, "data", "processed"))
