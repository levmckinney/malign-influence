# Modify NLTK's data path to use our custom directory
import nltk
from datasets import load_dataset
from nltk import downloader
from nltk.downloader import Downloader

nltk.data.path.insert(0, "/mfs1/u/max/.cache/nltk_data")

# Alternative approach: directly modify the global downloader instance
downloader._downloader = Downloader(download_dir="/mfs1/u/max/.cache/nltk_data")
downloader.download = downloader._downloader.download

subsets = {
    "algebraic-stack": "data/algebraic-stack/**/*-0002.json.gz",
    "arxiv": "data/arxiv/**/*-0002.json.gz",
    "dclm": "data/dclm/**/global-shard_01_of_10/local-shard_0_of_10/000000*.jsonl.zstd",
    "open-web-math": "data/open-web-math/**/*-0002.json.gz",
    "pes2o": "data/pes2o/**/*0002.json.gz",
    "starcoder": "data/starcoder/**/c-0002.json.gz",
    "wiki": "data/wiki/**/*-0001.json.gz",
}  # No dclm due to procesing issues


for config_name,path in subsets.items():
    if config_name in ["dclm", "starcoder"]:
        print(f"Processing {config_name}")
        dataset = load_dataset(
            "allenai/olmo-mix-1124", config_name, data_files={"train": path}
        )
        # Get the cache directory location
        print(dataset)
