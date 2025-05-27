"""
    Launch Slurm jobs pythonically.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

from omegaconf import OmegaConf

@dataclass
class StoolArgs:
    mlflow_uri: str = "localhost:5000"
    job_name: str = "toy"
    nodes: int = 1
    ngpu: int = 1
    ncpu: int = 16
    ntasks: int = 1
    mem: str = "63000"
    time: str = ""
    partition: str = "main"
    launcher: str = "sbatch"
    output_log: str = ""
    conda_env: str = ""
    job_dir: str = "./slurm-jobs/"

SBATCH_CMD = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --output={output_log}

export MLFLOW_TRACK_URI={mlflow_uri}
conda activate {conda_env}
"""

def launch_job(args: StoolArgs) -> None:
    # Just check here
    assert args.time != "", "No time specified for job."
    assert args.output_log != "", "No output log specified."
    assert args.conda_env != "", "No conda environment to activate."

    sbatch = SBATCH_CMD.format(
        job_name=args.job_name,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks,
        gpus=args.ngpu,
        cpus=args.ncpu,
        time=args.time,
        partition=args.partition,
        output=args.output_log,
        mlflow_uri=args.mlflow_uri,
        conda_env=args.conda_env,
        output_log=args.output_log,
    )
    # Store StoolArgs config somewhere to keep track of the experiment and
    # Slurm job script.
    Path(args.job_dir).mkdir(exist_ok=True,parents=True)
    slurm_file = Path(args.job_dir) / (args.job_name + '.slurm')
    with open(slurm_file, "w") as fi:
        fi.write(sbatch)

    print(sbatch)
    os.system(f"{args.launcher} {str(slurm_file)}")
    print("Launched job")

if __name__ == '__main__':
    args = OmegaConf.from_cli()

    # Build StoolArgs
    default = OmegaConf.structured(StoolArgs)

    # Override StoolArgs with CLI arguments.
    OmegaConf.set_struct(default, True)
    override = OmegaConf.create(args)
    args = OmegaConf.to_object(OmegaConf.merge(default, override))

    # Launch Job.
    launch_job(args)