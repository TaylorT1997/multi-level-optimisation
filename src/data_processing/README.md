# Data Processing

## create_data_dirs

### Usage
Creates the necessary data dirs for downloading and processing the datasets

### Flags
- -r --root: Specifies the root dir to create data dirs, "../../" (two levels above current dir) by default

## download_data

### Usage
Downloads each of the datasets used in this project to the "data/raw" directory and unzips them

### Flags
- -a --all: If used downloads all the datasets, False by default
- -d --data: Specifies particular datasets to download, empty by default
- -t --target: Specifies the target dir to download to, "../../data/raw" by default
- -c --clean: If used deletes any downloaded zip/tar files after unzipping, False by default

## process_data

### Usage
Processes the downloaded (and unzipped) raw dataset files and converts them into tsv format

### Flags
- -a --all: If used processes all the datasets, False by default
- -d --data: Specifies particular datasets to process, empty by default
- -t --target: Specifies the target dir to download to, "../../data/processed" by default
- -s --source: Specifies the target dir to download to, "../../data/raw" by default
- -c --clean: If used deletes any downloaded zip/tar files after unzipping, False by default
- -e --errors: If used includes the types of errors (if available), False by default