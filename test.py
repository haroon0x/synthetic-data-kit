import json
from pathlib import Path
import os
import requests

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.core.save_as import convert_format
from synthetic_data_kit.core.create import process_file

api_key = "AIzaSyCEix7-4qBHirMOKCg9pKDU9Lbax0QYtqs"
config_path = Path(r"E:\facet-ai\my_config.yaml")
output_dir  = r"E:\facet-ai\data\processed_datasets"
api_base="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


file_path = r"E:\facet-ai\data\output\bitcoin.txt"

output_file = process_file(
    file_path=file_path,
    output_dir= output_dir,
    config_path=config_path,
    api_base=api_base,
    model="gemini-2.0-flash",
    provider="api-endpoint",
    num_pairs=10,
    )



print(f"Generated QA pairs saved to: {output_file}")

hf_dataset_path = os.path.join(output_dir, "hf_dataset")
