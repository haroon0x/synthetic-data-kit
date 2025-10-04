import json
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.core.ingest import process_file as ingest
from synthetic_data_kit.core.create import process_file
from synthetic_data_kit.core.curate import curate_qa_pairs
from synthetic_data_kit.core.save_as import convert_format

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"] 
api_key = GEMINI_API_KEY
config_path = Path(r"E:\facet-ai\my_config.yaml")
output_dir  = r"E:\facet-ai\data\processed_datasets"
api_base="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
model = "gemini-2.5-flash"

file_path = r"E:\facet-ai\sample\bitcoin.pdf"



def synthetic_data_pipline(file_path, output_dir, config_path, api_base, model):
    """ Complete synthetic data generation pipeline from file ingestion to dataset creation
    
    Args:
        file_path: Path to the input file (e.g., PDF, DOCX, TXT)
        output_dir: Directory to save intermediate and final outputs
        config_path: Path to the configuration YAML file    
        api_base: API base URL for the LLM
        model: Model name to use for the LLM
    Returns:
        Path to the final dataset in Hugging Face DatasetDict format


    """
    file_name = file_path.split("\\")[-1]
    file_type = file_path.rsplit(".", 1)[1].lower()

    parsed_txt = ingest(#parse text from file
        file_path=file_path,
        output_dir= output_dir,
        output_name=file_name,
        )
    output_file = process_file( #create QA pairs from text file
        file_path=parsed_txt,
        output_dir= output_dir,
        config_path=config_path,
        api_base=api_base,
        model=model,
        provider="api-endpoint",
        num_pairs=10,
        )
    curated_output = curate_qa_pairs(#curate QA pairs using LLM
        input_path = output_file,
        output_path =  os.path.join(output_dir, f"{file_name}_curated.json"),
        api_base= api_base,
        model = model,
        config_path = config_path,
        verbose = True,
        provider="api-endpoint",
    )
    dataset_dict=  convert_format(#convert to Hugging Face dataset using chatml format and returns the datasetDict
        input_path = curated_output,
        output_path = output_dir,
        format_type = "chatml",
        storage_format= "hf",
    )

    
    return dataset_dict