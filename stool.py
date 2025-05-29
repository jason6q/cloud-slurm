"""
    Launch Slurm jobs pythonically. This will build out a command for you.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

from omegaconf import OmegaConf

@dataclass
class StoolArgs:
    config: str = ""
    file: str = ""
    mlflow_uri: str = "http://localhost:5000"
    job_name: str = "toy"
    nodes: int = 1
    ngpu: int = 1
    ncpu: int = 16
    ntasks: int = 1
    mem: str = "63000"
    time: str = "01:00:00"
    partition: str = "main"
    launcher: str = "sbatch"
    output_log: str = ""
    conda_env: str = ""
    job_dir: str = "./slurm-jobs/"
    kernprof: bool = True

SBATCH_CMD = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --output={output_log}

export MASTER_PORT=12355
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_PORT="$MASTER_PORT
echo "MASTER_ADDR="$MASTER_ADDR

export MLFLOW_TRACKING_URI={mlflow_uri}
export MLFLOW_UI_DEFAULT_CHART_TYPE=line

conda activate {conda_env}
"""

def launch_job(args: StoolArgs) -> None:
    """
        Build the launcher command and execute it.
    """
    # Just check here
    assert args.file != "", "Specify the file you want to execute."
    assert args.config != "", "Specify a config that contains all model specific parameters."
    assert args.output_log != "", "No output log specified."
    assert args.conda_env != "", "No conda environment to activate."

    sbatch = SBATCH_CMD.format(
        file=args.file,
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

    # Check if we are using kernel profiling
    # Need to make sure you are adding @profile decorators if you plan
    # on using it.
    if args.kernprof:
        sbatch = f"{sbatch}\n kernprof -o {args.output_log}.lprof -l {args.file}"

    # Store StoolArgs config somewhere to keep track of the experiment and
    # Slurm job script.
    Path(args.job_dir).mkdir(exist_ok=True, parents=True)
    slurm_file = Path(args.job_dir) / (args.job_name + '.slurm')
    with open(slurm_file, "w") as fi:
        fi.write(sbatch)

    # Launch
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