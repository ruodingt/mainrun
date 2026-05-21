import yaml
from pathlib import Path

def get_or_create_experiment_dir(args, base_dir: str = "./experiments") -> Path:
    """
    Resolves the expXXX directory based on hyperparameter fingerprint.
    Reuses existing expXXX if fingerprints match, otherwise creates a new one.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    fingerprint = args.get_fingerprint(ignore=['seed', 'log_file'])
    
    # Scan existing expXXX directories
    existing_exps = []
    if base_path.exists():
        for p in base_path.iterdir():
            if p.is_dir() and p.name.startswith("exp") and p.name[3:].isdigit():
                existing_exps.append(p)
                
    # Sort them by their numeric ID
    existing_exps.sort(key=lambda x: int(x.name[3:]))
    
    # Check if any experiment matches the current fingerprint
    for exp_path in existing_exps:
        hp_path = exp_path / "hp.yaml"
        if hp_path.exists():
            try:
                with open(hp_path, 'r') as f:
                    hp_data = yaml.safe_load(f)
                if hp_data == fingerprint:
                    return exp_path
            except Exception:
                pass
                
    # If no match, create a new expXXX directory
    if existing_exps:
        next_num = int(existing_exps[-1].name[3:]) + 1
    else:
        next_num = 0
        
    new_exp_name = f"exp{next_num:03d}"
    new_exp_path = base_path / new_exp_name
    new_exp_path.mkdir(parents=True, exist_ok=True)
    
    # Write hp.yaml
    hp_path = new_exp_path / "hp.yaml"
    with open(hp_path, 'w') as f:
        yaml.safe_dump(fingerprint, f, default_flow_style=False)
        
    return new_exp_path

def create_run_dir(exp_dir: Path, seed: int) -> Path:
    """
    Creates a new runYY subdirectory inside the specified exp_dir.
    Automatically increments the run number.
    """
    # Scan existing runYY directories
    existing_runs = []
    for p in exp_dir.iterdir():
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit():
            existing_runs.append(p)
            
    # Sort them by their numeric ID
    existing_runs.sort(key=lambda x: int(x.name[3:]))
    
    # Get next run number
    if existing_runs:
        next_num = int(existing_runs[-1].name[3:]) + 1
    else:
        next_num = 1
        
    new_run_name = f"run{next_num:02d}"
    new_run_path = exp_dir / new_run_name
    new_run_path.mkdir(parents=True, exist_ok=True)
    
    # Write run.yaml
    run_yaml_path = new_run_path / "run.yaml"
    with open(run_yaml_path, 'w') as f:
        yaml.safe_dump({"seed": seed}, f, default_flow_style=False)
        
    return new_run_path

def save_model_summary(model, exp_dir: Path, run_dir: Path):
    """
    Saves a comprehensive text summary of the model structure and parameters
    to both the expXXX and runYY directories.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    summary_content = (
        "=========================================\n"
        "Model Architecture\n"
        "=========================================\n"
        f"{model}\n\n"
        "=========================================\n"
        "Parameter Analysis\n"
        "=========================================\n"
        f"Total Parameters: {total_params:,}\n"
        f"Trainable Parameters: {trainable_params:,}\n"
        f"Non-trainable Parameters: {non_trainable_params:,}\n"
    )
    
    # Write to expXXX/model_summary.txt
    exp_summary_path = exp_dir / "model_summary.txt"
    with open(exp_summary_path, 'w') as f:
        f.write(summary_content)
        
    # Write to runYY/model_summary.txt
    run_summary_path = run_dir / "model_summary.txt"
    with open(run_summary_path, 'w') as f:
        f.write(summary_content)
