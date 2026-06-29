#!/usr/bin/env python3
"""
GPT Description Generator

This script generates descriptions for drugs or cell lines using Azure OpenAI's GPT models.
It processes items in batches and saves the results to CSV files with configurable parameters.
openai==0.28.0 is required for this script to run.
"""
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal
from tqdm import tqdm
import pandas as pd
import openai


class ConfigManager:
    """Manages configuration and environment variables for the application."""    
    def __init__(self):
        # Resolve paths from this script location
        self.base_dir = Path(__file__).resolve().parent
        self.data_dir = self.base_dir.parent / 'data_example'
        self.prompt_dir = self.base_dir.parent / 'prompts'
        self.result_dir = self.base_dir.parent / 'result'
        
        # Configure OpenAI
        self._setup_openai()
        
        # Create necessary directories
        self._setup_directories()
    
    def _setup_openai(self) -> None:
        """Configure OpenAI API settings from environment variables."""
        required_vars = [
            "AZURE_OPENAI_KEY",
            "AZURE_OPENAI_ENDPOINT"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
            
        openai.api_type = "azure"
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_version = "2023-09-15-preview"
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [
            self.result_dir / 'drug' / 'desc',
            self.result_dir / 'cell' / 'desc'
        ]:
            directory.mkdir(parents=True, exist_ok=True)


class DescriptionGenerator:
    """Handles the generation of descriptions"""    
    def __init__(
        self,
        config: ConfigManager,
        model, 
        temperature: float = 0.2,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        top_p: float = 0.95
    ):
        self.model = model
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.top_p = top_p
        self.config = config
    
    def generate_description(
        self,
        name: str,
        prompt_template: str,
        type_: Literal["drug", "cell"]
    ) -> str:
        """
        Args:
            name: Name of the drug or cell line
            prompt_template: Template string for the prompt
            type_: Type of description to generate ("drug" or "cell")
            
        Returns:
            Generated description as a string
        """
        try:
            # Format prompt based on type
            query_data = {
                type_: name,
            }
            prompt = prompt_template.format(**query_data)
            
            response = openai.Completion.create(
                engine=self.model, # Azure deployment name
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=300,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                best_of=1,
                stop=None
            )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            print(f"Error generating description for {name}: {str(e)}")
            return f"Error: Failed to generate description for {name}"
    
    def process_items(
        self,
        items: List[str],
        prompt_template: str,
        type_: Literal["drug", "cell"]
    ) -> pd.DataFrame:
        """
        Args:
            items: List of drug or cell line names
            prompt_template: Template string for the prompt
            type_: Type of items to process ("drug" or "cell")
            
        Returns:
            DataFrame containing items and their descriptions
        """
        descriptions = []
        
        for item in tqdm(items, desc=f"Generating {type_} descriptions"):
            description = self.generate_description(item, prompt_template, type_)
            descriptions.append(description)
        
        df = pd.DataFrame({
            type_: items,
            'description': descriptions
        })
                
        return df
    
    def save_results(
        self,
        df: pd.DataFrame,
        type_: Literal["drug", "cell"]
    ) -> None:
        """Save results to a CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"generated_desc_{type_}.csv")
        
        output_dir = (self.config.result_dir / type_ / 'desc')
        output_path = output_dir / filename
        
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate descriptions for drugs or cell lines using GPT"
    )
    parser.add_argument("--model", type=str, required=True,
                      help="GPT model to use")
    parser.add_argument("--temp", type=float, default=0.2,
                      help="Temperature for GPT generation")
    parser.add_argument("--freq", type=float, default=0,
                      help="Frequency penalty for GPT generation")
    parser.add_argument("--presence", type=float, default=0,
                      help="Presence penalty for GPT generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                      help="Top p sampling parameter")
    parser.add_argument("--prompt", type=str, default=None, 
                      help="Prompt template file name (defaults to prompt_<type>.txt)")
    parser.add_argument("--data", type=str, required=True,
                      help="Name of the dataset")
    parser.add_argument("--type", type=str, required=True, choices=["drug", "cell"],
                      help="drug or cell")
    args = parser.parse_args()

    print("Starting description generation process")
    config = ConfigManager()

    # Initialize generator
    generator = DescriptionGenerator(
        config=config,
        model=args.model,
        temperature=args.temp,
        frequency_penalty=args.freq,
        presence_penalty=args.presence,
        top_p=args.top_p,
    )
    
    data_ = args.data
    type_ = args.type.lower()

    prompt_file = args.prompt if args.prompt is not None else f"prompt_{type_}.txt"

    try:
        with open(config.prompt_dir / prompt_file) as f:
            prompt_template = f.read().strip()
    except FileNotFoundError as e:
        print(f"Could not find required file: {e.filename}")
        raise
    
    df = pd.read_csv(config.data_dir / f'{data_}_{type_}.csv')
    items = df[type_].dropna().astype(str).tolist()
    
    results_df = generator.process_items(items, prompt_template, type_)
    generator.save_results(results_df, type_)
    
    print("Finished!")