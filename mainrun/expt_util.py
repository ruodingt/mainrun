import json
import time
from pathlib import Path

import structlog
import yaml
from tqdm import tqdm


def get_or_create_experiment_dir(args, base_dir: str = "./experiments") -> Path:
    """
    Resolves the expXXX directory based on hyperparameter fingerprint.
    Reuses existing expXXX if fingerprints match, otherwise creates a new one.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    fingerprint = args.get_fingerprint()

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


def create_run_dir(exp_dir: Path, args) -> Path:
    """
    Creates a new runYY subdirectory inside the specified exp_dir.
    Saves full hyperparameter snapshot to run.yaml (results added later via save_run_results).
    """
    existing_runs = sorted(
        [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit()],
        key=lambda x: int(x.name[3:])
    )
    next_num = int(existing_runs[-1].name[3:]) + 1 if existing_runs else 1

    run_path = exp_dir / f"run{next_num:02d}"
    run_path.mkdir(parents=True, exist_ok=True)

    import dataclasses
    all_params = dataclasses.asdict(args)
    all_params.pop('log_file', None)
    with open(run_path / "run.yaml", 'w') as f:
        yaml.safe_dump(all_params, f, default_flow_style=False, sort_keys=True)

    return run_path


def save_run_results(run_dir: Path, val_loss: float, total_time_s: float,
                     avg_tok_s: float = 0.0, total_params: int = 0):
    """Append final training results to run.yaml."""
    run_yaml_path = run_dir / "run.yaml"
    with open(run_yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    data['results'] = {
        'val_loss': round(val_loss, 6),
        'total_time_s': round(total_time_s, 1),
        'avg_tok_s': round(avg_tok_s, 1),
        'total_params': total_params,
    }
    with open(run_yaml_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)


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


def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_handler = open(log_file, 'w')

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    class DualLogger:
        def __init__(self, file_handler):
            self.file_handler = file_handler
            self.logger = structlog.get_logger()

        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()

            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(
                        f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s")
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)

    return DualLogger(file_handler)
