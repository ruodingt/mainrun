from datasets import load_dataset
import sys, logging, datasets


# Enable HTTP request logging
logging.basicConfig(level=logging.DEBUG)
datasets.logging.set_verbosity_debug()

# Enable requests/urllib3 logging for network details
import requests
import http.client as http_client

http_client.HTTPConnection.debuglevel = 1

# Also enable urllib3 logging
logging.getLogger("requests").setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.DEBUG)

print("Downloading Hacker News dataset...")
try:
    # Download and cache the dataset
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    sys.exit(1)

# HF_ENDPOINT=https://hf-mirror.com python download_dataset.py
