from datasets import load_dataset
from nltk import downloader
from nltk.downloader import Downloader
from dolma.cli.tokenizer import TokenizerCli, TokenizationConfig

# Modify NLTK's data path to use our custom directory
import nltk
nltk.data.path.insert(0, "/mfs1/u/max/.cache/nltk_data")

# Alternative approach: directly modify the global downloader instance
downloader._downloader = Downloader(download_dir="/mfs1/u/max/.cache/nltk_data")
downloader.download = downloader._downloader.download
dataset = load_dataset("allenai/olmo-mix-1124", split="train[0:20]")
# Get the cache directory location
cache_dir = dataset.cache_files() # type: ignore
print(f"Dataset cache location: {cache_dir}")

tokenizer_config = TokenizationConfig(
    documents=cache_dir,
    destination="dolma-tokenized-dataset",
    tokenizer_name_or_path="allenai/OLMo-2-1124-7B-Instruct",
)

tokenizer_cli = TokenizerCli.run(tokenizer_config)
