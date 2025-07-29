"""
MatBench dataset setup and analysis
Focus on: formation energy, band gap, bulk modulus for Day 2
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import subprocess
import sys
import os

try:
    from matbench.bench import MatbenchBenchmark
    MATBENCH_AVAILABLE = True
except ImportError:
    print("MatBench not installed. Attempting to install...")

    # Detect if we're in a Conda environment
    in_conda_env = os.getenv("CONDA_PREFIX") is not None

    if in_conda_env:
        try:
            subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "matbench"])
            from matbench.bench import MatbenchBenchmark
            MATBENCH_AVAILABLE = True
        except subprocess.CalledProcessError:
            print("❌ Failed to install matbench using conda.")
            sys.exit(1)
    else:
        print("❌ Not in a conda environment. Please install matbench manually using:")
        print("    pip install matbench  # or better, use conda")
        sys.exit(1)

class MatBenchAnalyzer:
    """Analyze MatBench datasets for materials property prediction"""
    
    PRIORITY_TASKS = [
        'matbench_mp_e_form',     # Formation energy
        'matbench_mp_gap',        # Band gap  
        'matbench_mp_bulk_modulus', # Bulk modulus
        'matbench_steels',        # Steel yield strength
    ]
    
    def __init__(self, data_dir: str = "data/raw/matbench"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mb = MatbenchBenchmark(autoload=False)
        
    def analyze_task(self, task_name: str) -> Dict:
        """Analyze specific MatBench task"""
        print(f"\nAnalyzing {task_name}...")
        
        try:
            # Load task data
            task = getattr(self.mb, task_name)
            df = task.df
            
            # Basic analysis
            analysis = {
                'task_name': task_name,
                'size': len(df),
                'input_type': 'structure' if 'structure' in df.columns else 'composition',
                'target_col': [col for col in df.columns if col not in ['structure', 'composition']][0]
            }
            
            # Target analysis
            target_col = analysis['target_col']
            target_data = df[target_col].dropna()
            
            analysis.update({
                'target_mean': target_data.mean() if target_data.dtype in [np.float64, np.int64] else None,
                'target_std': target_data.std() if target_data.dtype in [np.float64, np.int64] else None,
                'target_min': target_data.min() if target_data.dtype in [np.float64, np.int64] else None,
                'target_max': target_data.max() if target_data.dtype in [np.float64, np.int64] else None,
                'missing_values': df[target_col].isna().sum(),
                'unique_values': df[target_col].nunique()
            })
            
            # Save processed data
            output_path = self.data_dir / f"{task_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
            
            return analysis, df
            
        except Exception as e:
            print(f"Error analyzing {task_name}: {e}")
            return None, None
    
    def create_summary_report(self, analyses: List[Dict]):
        """Create comprehensive analysis report"""
        summary_df = pd.DataFrame(analyses)
        
        # Save summary
        summary_path = Path("data/processed/matbench/task_summary.csv")  
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        
        return summary_df

def main():
   """Analyze priority MatBench tasks"""
   analyzer = MatBenchAnalyzer()
   
   analyses = []
   datasets = {}
   
   print("=== MatBench Dataset Analysis ===")
   
   for task_name in analyzer.PRIORITY_TASKS:
       analysis, df = analyzer.analyze_task(task_name)
       
       if analysis is not None:
           analyses.append(analysis)
           datasets[task_name] = df
           
           print(f"✅ {task_name}: {analysis['size']} samples")
           if analysis['target_mean'] is not None:
               print(f"   Target range: [{analysis['target_min']:.3f}, {analysis['target_max']:.3f}]")
       else:
           print(f"❌ Failed to load {task_name}")
   
   # Create summary report
   if analyses:
       summary_df = analyzer.create_summary_report(analyses)
       print(f"\n📊 MatBench Summary:")
       print(summary_df[['task_name', 'size', 'input_type']].to_string(index=False))
       
       # Save to results for Day 2 analysis
       results_path = Path("results/day02/matbench_analysis.txt")
       results_path.parent.mkdir(parents=True, exist_ok=True)
       
       with open(results_path, 'w') as f:
           f.write("=== MatBench Analysis Results ===\n")
           f.write(f"Analysis date: {pd.Timestamp.now()}\n\n")
           f.write(summary_df.to_string(index=False))
   
   return datasets, analyses

if __name__ == "__main__":
   datasets, analyses = main()
