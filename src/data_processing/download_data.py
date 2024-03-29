import sys
import os
import argparse
import requests
import zipfile
import tarfile
import pandas as pd


def download_from_url(url, target_path):
    r = requests.get(url)

    with open(target_path, "wb") as f:
        f.write(r.content)


def unzip_file(zip_path, target_path, cleanup):
    if zip_path.endswith("tar.gz"):
        with tarfile.open(zip_path, "r:gz") as tar:
            tar.extractall(target_path)
    elif zip_path.endswith("tar"):
        with tarfile.open(zip_path, "r:") as tar:
            tar.extractall(target_path)
    elif zip_path.endswith("zip"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_path)

    if cleanup:
        os.remove(zip_path)


if __name__ == "__main__":
    dataset_names = ["fce", "conll_10", "wi_locness"]

    parser = argparse.ArgumentParser(description="Dataset downloader and unzipper")
    parser.add_argument(
        "-a", "--all", action="store_true", help="Download all datasets"
    )
    parser.add_argument(
        "-d",
        "--data",
        action="store",
        nargs="+",
        default=[],
        help="Download specific datasets:{}".format(dataset_names),
    )
    parser.add_argument(
        "-t",
        "--target",
        action="store",
        default="../../data/raw",
        help="Target directory to download to",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        default=False,
        help="Removes any downloaded tar/zip files",
    )

    args = parser.parse_args()

    download_all = args.all
    download_datasets = args.data
    target_path = args.target
    cleanup = args.clean

    if download_all or "fce" in download_datasets:
        url = (
            "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz"
        )

        download_from_url(url, os.path.join(target_path, "fce.tar.gz"))
        unzip_file(
            os.path.join(target_path, "fce.tar.gz"),
            os.path.join(target_path, "fce_v2.1"),
            cleanup,
        )

    if download_all or "conll_10" in download_datasets:
        url = (
            "http://www.inf.u-szeged.hu/~vinczev/conll2010st/task2_train_bio_rev2_1.zip"
        )

        download_from_url(url, os.path.join(target_path, "conll_10.zip"))
        unzip_file(
            os.path.join(target_path, "conll_10.zip"),
            os.path.join(target_path, "conll_10"),
            cleanup,
        )

        url = "http://www.inf.u-szeged.hu/~vinczev/conll2010st/task2_eval_rev.zip"

        download_from_url(url, os.path.join(target_path, "conll_10.zip"))
        unzip_file(
            os.path.join(target_path, "conll_10.zip"),
            os.path.join(target_path, "conll_10"),
            cleanup,
        )

    if download_all or "wi_locness" in download_datasets:
        url = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"

        download_from_url(url, os.path.join(target_path, "wi_locness_v2.1.tar.gz"))
        unzip_file(
            os.path.join(target_path, "wi_locness_v2.1.tar.gz"),
            os.path.join(target_path, "wi_locness_v2.1"),
            cleanup,
        )
