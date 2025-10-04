import json
from pathlib import Path
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.core.save_as import convert_format
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.curate import curate_qa_pairs

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"] 
api_key = GEMINI_API_KEY
config_path = Path(r"E:\facet-ai\my_config.yaml")
output_dir  = r"E:\facet-ai\data\processed_datasets"
api_base="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
model = "gemini-2.5-flash"

file_path = r"E:\facet-ai\data\output\bitcoin.txt"
"""
output_file = process_file(
    file_path=file_path,
    output_dir= output_dir,
    config_path=config_path,
    api_base=api_base,
    model=model,
    provider="api-endpoint",
    num_pairs=10,
    )"""
print(f"Generated QA pairs saved to: ")

curated_output = curate_qa_pairs(
    input_path = r"E:\facet-ai\data\processed_datasets\bitcoin_qa_pairs.json",
    output_path =  os.path.join(output_dir, "bitcoin_qa_pairs_curated.json"),
    api_base= api_base,
    model = model,
    config_path = config_path,
    verbose = True,
    provider="api-endpoint",
)

dataset_path =  convert_format(
    input_path = curated_output,
    output_path = output_dir,
    format_type = "chatml",
    storage_format= "hf",
)
print(f"dataset saved to: {dataset_path}")