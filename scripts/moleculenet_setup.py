"""
MoleculeNet dataset acquisition and analysis
Focus on: ESOL, FreeSolv, Lipophilicity for Day 2
"""
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import urllib.request
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class MoleculeNetDownloader:
    """Download and analyze MoleculeNet datasets"""
    
    #BASE_URL = "https://deepchem.io/datasets/"
    DATASETS = {
        'esol': {
            'url': 'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv',
            'target_col': 'measured log solubility in mols per litre',
            'smiles_col': 'smiles',
            'task_type': 'regression',
            'description': 'Water solubility prediction'
        },
        'freesolv': {
            'url': 'https://github.com/MobleyLab/FreeSolv/raw/master/database.txt',
            'target_col': 'expt',
            'smiles_col': 'smiles', 
            'task_type': 'regression',
            'description': 'Hydration free energy prediction'
        },
        'lipophilicity': {
            'url': 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
            'target_col': 'exp',
            'smiles_col': 'smiles',
            'task_type': 'regression', 
            'description': 'Lipophilicity prediction'
        }
    }
    
    def __init__(self, data_dir: str = "data/raw/moleculenet"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Download specific dataset"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported")
            
        dataset_info = self.DATASETS[dataset_name]
        filepath = self.data_dir / f"{dataset_name}.csv"
        
        print(f"Downloading {dataset_name}...")
        
        try:
            if dataset_name == 'freesolv':
                # Special handling for FreeSolv format
                df = pd.read_csv(dataset_info['url'], sep=';', comment='#')
                print("cabecita", df.head())
            else:
                df = pd.read_csv(dataset_info['url'])
                
            df.to_csv(filepath, index=False)
            print(f"Saved to {filepath}")
            return df
            
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return None
    
    def analyze_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Analyze dataset characteristics"""
        info = self.DATASETS[dataset_name]
        
        analysis = {
            'name': dataset_name,
            'description': info['description'],
            'size': len(df),
            'task_type': info['task_type'],
            'target_col': info['target_col'],
            'smiles_col': info['smiles_col']
        }
        
        # Target analysis
        if info['target_col'] in df.columns:
            target_data = df[info['target_col']].dropna()
            analysis.update({
                'target_mean': target_data.mean(),
                'target_std': target_data.std(),
                'target_min': target_data.min(),
                'target_max': target_data.max(),
                'missing_targets': df[info['target_col']].isna().sum()
            })
        
        # SMILES analysis
        if info['smiles_col'] in df.columns:
            smiles_data = df[info['smiles_col']].dropna()
            analysis.update({
                'avg_smiles_length': smiles_data.str.len().mean(),
                'missing_smiles': df[info['smiles_col']].isna().sum(),
                'unique_smiles': smiles_data.nunique()
            })
        
        return analysis

def main():
    """Download and analyze all MoleculeNet datasets"""
    downloader = MoleculeNetDownloader()
    
    analyses = []
    datasets = {}
    
    # Download priority datasets for Day 2
    priority_datasets = ['freesolv', 'esol', 'lipophilicity']
    
    for dataset_name in priority_datasets:
        print(f"\n=== Processing {dataset_name.upper()} ===")
        df = downloader.download_dataset(dataset_name)
        
        if df is not None:
            datasets[dataset_name] = df
            analysis = downloader.analyze_dataset(df, dataset_name)
            analyses.append(analysis)
            
            print(f"Dataset size: {analysis['size']}")
            print(f"Task: {analysis['description']}")
            if 'target_mean' in analysis:
                print(f"Target range: [{analysis['target_min']:.3f}, {analysis['target_max']:.3f}]")
    
    # Create summary analysis
    summary_df = pd.DataFrame(analyses)
    summary_path = Path("data/processed/moleculenet/dataset_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n✅ Summary saved to {summary_path}")
    print("\n📊 Dataset Summary:")
    print(summary_df[['name', 'size', 'description']].to_string(index=False))
    
    return datasets, analyses

if __name__ == "__main__":
    datasets, analyses = main()
