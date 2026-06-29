#!/usr/bin/env python3

import openai
import os
import argparse
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import pickle
import pandas as pd
from typing import List


class ConfigManager:
    """Manages configuration and environment variables for embedding generation."""
    def __init__(self):
        self.api_type = "azure"
        self.api_version = "2024-06-01"
        self.embedding_engine = "embed-ada-2"  # deployed embedding model name

        self.base_dir = Path(__file__).resolve().parent
        self.result_dir = self.base_dir.parent / 'result'
        self.result_dirs = {
            'drug': {
                'desc': self.result_dir / 'drug' / 'desc',
                'embed': self.result_dir / 'drug' / 'embed',
            },
            'cell': {
                'desc': self.result_dir / 'cell' / 'desc',
                'embed': self.result_dir / 'cell' / 'embed',
            }
        }

        self._setup_openai()
        self._setup_directories()

    def _setup_openai(self) -> None:
        required_vars = [
            "AZURE_OPENAI_KEY",
            "AZURE_OPENAI_ENDPOINT"
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        openai.api_type = self.api_type
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_version = self.api_version

    def _setup_directories(self) -> None:
        """Create required output directories if they don't exist."""
        for directory in [
            self.result_dirs['drug']['embed'],
            self.result_dirs['cell']['embed'],
        ]:
            directory.mkdir(parents=True, exist_ok=True)


class EmbeddingGenerator:
    """Class to handle embedding generation"""
    def __init__(self, config: ConfigManager, type_: str):
        self.config = config
        self.type_ = type_
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if type_ not in self.config.result_dirs:
            raise ValueError(f"Invalid filetype: {type_}. Expected 'drug' or 'cell'")
        self.dirs = self.config.result_dirs[type_]

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text"""
        try:
            response = openai.Embedding.create(
                engine=self.config.embedding_engine,
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

    def process_data(self, dat, dat_type):
        embed = []

        for idx, row in tqdm(dat.iterrows(), total=len(dat), desc=f"Getting Embeddings of {dat_type}s"):
            item = row['description']

            if pd.isna(item):
                item = ""
            else:
                item = str(item).replace("\\n", " ").replace("\n", " ").strip()

            embed.append(self.get_embedding(item))
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} items. Pausing for 0.5 minute...")
                time.sleep(30)
                print("Resuming processing...")
                
        names = dat[dat_type].astype(str).tolist()
        embed_dict = dict(zip(names, embed))
        return embed_dict, embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for descriptions")
    parser.add_argument("--filename", type=str, required=True, 
                       help="Filename of generated descriptions csv which contains 'description' column")
    parser.add_argument("--type", type=str, required=True, 
                       choices=['drug', 'cell'], help="Specify drug or cell")
    args = parser.parse_args()

    print("Starting embedding generation...")

    try:
        type_ = args.type.lower()
        config = ConfigManager()
        analyzer = EmbeddingGenerator(config=config, type_=type_)

        dat = pd.read_csv(analyzer.dirs['desc'] / args.filename)
        embed_dict, _ = analyzer.process_data(dat, type_)

        filepath = analyzer.dirs['embed'] / f'embed_gpt_{type_}_{analyzer.timestamp}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(embed_dict, f)

        print(f"Process completed: Embeddings saved to {filepath}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise