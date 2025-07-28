# ============================================
# ANALYZE DORA TEMPLATES FOR SCIENTIFIC ADAPTATION
# Save as: ~/peft-scientific/analyze_templates.py
# ============================================

import os
import glob
import re
from pathlib import Path

def analyze_dora_templates():
    """Analyze DoRA repo structure to find best template for scientific adaptation"""
    
    print("🔍 Analyzing DoRA Templates for Scientific Adaptation")
    print("=" * 60)
    
    repo_path = Path("~/peft-scientific/DoRA").expanduser()
    if not repo_path.exists():
        print(f"❌ DoRA repository not found at {repo_path}")
        return
    
    # 1. Analyze directory structure
    directories = [d for d in repo_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"📁 Available directories:")
    for i, dir_path in enumerate(directories, 1):
        file_count = len(list(dir_path.glob("*.py")))
        print(f"   {i}. {dir_path.name} ({file_count} Python files)")
    
    # 2. Deep analysis of each directory
    template_analysis = {}
    
    for dir_path in directories:
        if dir_path.name in ['.git', '__pycache__', '.pytest_cache']:
            continue
            
        analysis = analyze_directory(dir_path)
        if analysis['python_files'] > 0:  # Only include directories with Python files
            template_analysis[dir_path.name] = analysis
    
    # 3. Rank templates by suitability for scientific adaptation
    print(f"\n📊 TEMPLATE ANALYSIS:")
    print("-" * 80)
    print(f"{'Directory':<25} {'Files':<6} {'Size':<8} {'Complexity':<10} {'Suitability':<12}")
    print("-" * 80)
    
    ranked_templates = []
    
    for name, analysis in template_analysis.items():
        suitability_score = calculate_suitability_score(analysis)
        ranked_templates.append((name, analysis, suitability_score))
        
        print(f"{name:<25} {analysis['python_files']:<6} {analysis['total_lines']:<8} {analysis['complexity']:<10} {suitability_score:<12}")
    
    # Sort by suitability score
    ranked_templates.sort(key=lambda x: x[2], reverse=True)
    
    # 4. Detailed analysis of top 2 templates
    print(f"\n🏆 TOP TEMPLATES FOR SCIENTIFIC ADAPTATION:")
    
    for i, (name, analysis, score) in enumerate(ranked_templates[:2], 1):
        print(f"\n{i}. {name.upper()} (Score: {score})")
        print(f"   📁 Files: {analysis['python_files']} Python files")
        print(f"   📏 Size: {analysis['total_lines']} total lines")
        print(f"   🔧 Key features:")
        
        for feature in analysis['key_features']:
            print(f"      - {feature}")
        
        print(f"   📝 Main files:")
        for file_info in analysis['main_files'][:3]:
            print(f"      - {file_info['name']} ({file_info['lines']} lines)")
    
    # 5. Recommendation
    best_template = ranked_templates[0][0] if ranked_templates else None
    
    if best_template:
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   ✅ Use '{best_template}' as base template")
        print(f"   🔄 Adaptation strategy:")
        
        adaptation_strategy = get_adaptation_strategy(best_template, template_analysis[best_template])
        for step in adaptation_strategy:
            print(f"      {step}")
    
    return ranked_templates

def analyze_directory(dir_path):
    """Analyze a single directory for template suitability"""
    
    analysis = {
        'python_files': 0,
        'total_lines': 0,
        'main_files': [],
        'key_features': [],
        'complexity': 'Unknown'
    }
    
    # Count Python files and lines
    for py_file in dir_path.glob("**/*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                analysis['total_lines'] += lines
                analysis['python_files'] += 1
                
                analysis['main_files'].append({
                    'name': py_file.name,
                    'lines': lines,
                    'path': str(py_file.relative_to(dir_path))
                })
        except Exception:
            continue
    
    # Sort files by line count
    analysis['main_files'].sort(key=lambda x: x['lines'], reverse=True)
    
    # Detect key features
    features = detect_key_features(dir_path)
    analysis['key_features'] = features
    
    # Determine complexity
    if analysis['total_lines'] < 500:
        analysis['complexity'] = 'Simple'
    elif analysis['total_lines'] < 2000:
        analysis['complexity'] = 'Moderate'
    else:
        analysis['complexity'] = 'Complex'
    
    return analysis

def detect_key_features(dir_path):
    """Detect key features in a directory"""
    features = []
    
    # Check for common patterns
    all_content = ""
    for py_file in dir_path.glob("**/*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                all_content += f.read().lower()
        except Exception:
            continue
    
    # Feature detection patterns
    patterns = {
        'HuggingFace Integration': r'from transformers import|AutoModel|AutoTokenizer',
        'DoRA Implementation': r'use_dora|dora|DoRA',
        'Training Loop': r'def train|optimizer|loss\.backward',
        'Data Loading': r'DataLoader|Dataset|load_dataset',
        'Evaluation': r'def evaluate|def test|accuracy|bleu|rouge',
        'Configuration': r'argparse|config|ConfigDict',
        'Multi-GPU Support': r'accelerate|DeepSpeed|distributed',
        'Quantization': r'BitsAndBytes|quantiz|4bit|8bit',
        'Logging': r'wandb|tensorboard|logging',
        'Model Saving': r'save_pretrained|checkpoint|save_model'
    }
    
    for feature_name, pattern in patterns.items():
        if re.search(pattern, all_content):
            features.append(feature_name)
    
    return features

def calculate_suitability_score(analysis):
    """Calculate suitability score for scientific adaptation"""
    score = 0
    
    # Base score from file count (more files = more comprehensive)
    score += min(analysis['python_files'] * 10, 50)
    
    # Complexity bonus (moderate complexity is ideal)
    if analysis['complexity'] == 'Moderate':
        score += 30
    elif analysis['complexity'] == 'Simple':
        score += 20
    elif analysis['complexity'] == 'Complex':
        score += 10
    
    # Feature bonuses
    key_features = analysis['key_features']
    feature_bonuses = {
        'HuggingFace Integration': 25,
        'DoRA Implementation': 30,
        'Training Loop': 20,
        'Data Loading': 15,
        'Evaluation': 15,
        'Configuration': 10,
        'Multi-GPU Support': 5,
        'Quantization': 10,
        'Logging': 5,
        'Model Saving': 10
    }
    
    for feature in key_features:
        score += feature_bonuses.get(feature, 0)
    
    return score

def get_adaptation_strategy(template_name, analysis):
    """Get specific adaptation strategy for the chosen template"""
    
    strategies = {
        'commonsense_reasoning': [
            "1. Copy commonsense_reasoning/ to scientific_domains/",
            "2. Replace CommonsenseQA dataset with Materials Science dataset",
            "3. Adapt evaluation metrics (accuracy → BLEU/ROUGE)",
            "4. Modify data preprocessing for scientific text",
            "5. Update model configuration for longer sequences",
            "6. Add scientific vocabulary analysis"
        ],
        'instruction_tuning_dvora': [
            "1. Copy instruction_tuning/ to scientific_domains/",
            "2. Replace Alpaca dataset with scientific instruction data",
            "3. Adapt conversation format for scientific QA",
            "4. Modify evaluation for domain-specific tasks",
            "5. Update hyperparameters for scientific content",
            "6. Add cross-domain evaluation"
        ],
        'visual_instruction_tuning': [
            "1. Adapt visual components for scientific diagrams",
            "2. Replace visual datasets with scientific figures",
            "3. Modify text processing for scientific papers",
            "4. Add multi-modal scientific evaluation",
            "5. Update model architecture for sci-text",
            "6. Implement domain-specific routing"
        ]
    }
    
    return strategies.get(template_name, [
        "1. Analyze template structure carefully",
        "2. Identify core components to adapt",
        "3. Replace datasets with scientific data",
        "4. Modify evaluation metrics",
        "5. Update configuration files",
        "6. Test with small-scale experiments"
    ])

if __name__ == "__main__":
    print("🚀 Analyzing DoRA templates...")
    os.chdir(os.path.expanduser("~/peft-scientific"))
    
    templates = analyze_dora_templates()
    
    if templates:
        print(f"\n✅ Template analysis complete!")
        print(f"🎯 Recommended next action: Set up scientific adaptation")
    else:
        print(f"\n❌ No suitable templates found")
