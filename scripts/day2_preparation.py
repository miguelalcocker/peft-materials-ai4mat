"""
Day 2 preparation script - Queue setup tasks for efficient start
"""
import os
from pathlib import Path

def create_day2_structure():
    """Create directory structure for Day 2 data work"""
    directories = [
        "data/raw/moleculenet",
        "data/raw/matbench", 
        "data/processed/moleculenet",
        "data/processed/matbench",
        "experiments/baselines",
        "experiments/peft",
        "results/day02"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def create_data_download_script():
    """Create script for efficient data downloading"""
    script_content = '''#!/bin/bash
# Day 2 data download script
echo "Starting Day 2 data download..."

# Create data directories
mkdir -p data/raw/{moleculenet,matbench}

# MoleculeNet datasets (will implement in Day 2)
echo "MoleculeNet download prepared"

# MatBench datasets (will implement in Day 2) 
echo "MatBench download prepared"

echo "Day 2 data structure ready!"
'''
    
    with open("scripts/download_day2_data.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("scripts/download_day2_data.sh", 0o755)
    print("Created: scripts/download_day2_data.sh")

if __name__ == "__main__":
    print("=== Day 2 Preparation ===")
    create_day2_structure()
    create_data_download_script()
    
    print("\n✅ Day 2 preparation complete!")
    print("📋 Tomorrow's first task: Run ./scripts/download_day2_data.sh")
