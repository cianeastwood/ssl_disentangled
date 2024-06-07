"""
Submitit batch jobs.
"""

import argparse
import shlex
import submitit
import tqdm
from submitit.helpers import RsyncSnapshot
from utils import combine_argparse_yaml_args, create_temp_dir
from distributed import Trainer, get_init_file
from main import main as train_function
from main import get_arguments
from main_pretrained import main as train_function_pretrained
from main_pretrained import get_arguments as get_arguments_pretrained
import re


parser = argparse.ArgumentParser()
parser.add_argument('--commands_file', '-c', type=str, required=True)
parser.add_argument('--run-script', '-s', type=str, default='main.py')
parser.add_argument('--is-multi-gpu-job', action='store_true',
                    help="Imagenet requires more gpus and memory.")

# SLURM-specific paremters
parser.add_argument('--partition', '-p', type=str, default='devlab,learnlab,learnfair')
parser.add_argument('--slurmdir', '-sd', type=str, default='/checkpoint/ceastwood/jobs')
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--cpus_per_task', '-nc', type=int, default=8)
parser.add_argument('--gpus_per_node', '-ng', type=int, default=1)
parser.add_argument('--mem-per-gpu', '-mem', type=int)
parser.add_argument('--time', '-t', type=int, help="Maximum job time in minutes.")
parser.add_argument('--timeout_min', '-tom', type=int, default=4000, help="Minimum job time in minutes.")
parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
parser.add_argument('--comment', type=str, help='Comment to pass to scheduler, e.g. priority message')
parser.add_argument('--signal-delay-s', type=int, default=120,
                    help='Delay between the kill signal and the actual kill of the slurm job.')

args = parser.parse_args()

argparse_defaults = {k: parser.get_default(k) for k in vars(args).keys()}
jobname = args.commands_file.split("/")[-1].split(".")[0]                       # ignore .txt/.sh extension, e.g. reproduce_imagenet
dataset, sweepname = jobname.split("-")                                 # e.g. "reproduce" and "imagenet"
print(args.commands_file, jobname, dataset, sweepname)

# If imagenet, load dataset-specific slurm settings from .yaml config file (HACKY!)
args.is_multi_gpu_job = dataset in ["imagenet", "imagenet_blurred", "trifeature", "colordsprites"]
if args.is_multi_gpu_job:
    print(f"Submitting heavier multi-gpu jobs for {dataset}...")
    config_dataset = dataset.replace("_blurred", "")               # imagenet and imagenet_blurred share the same config
    combine_argparse_yaml_args(args, argparse_defaults, ["dataset"], f"configs/{config_dataset}/")
print("Args:", args)

# Load commands from rgs.commands_file file with one command per line
f = open(args.commands_file, "r")
cmds = f.read().split("\n")
n_jobs = len(cmds)
print(f'Submitting {n_jobs} jobs.')

# Prep kwargs
kwargs = {}
if args.use_volta32:
    kwargs['slurm_constraint'] = 'volta32gb'
if args.comment is not None:
    kwargs['slurm_comment'] = args.comment
if args.mem_per_gpu is not None:
    kwargs['mem_gb'] = args.mem_per_gpu * args.gpus_per_node
if args.time is not None:
    kwargs['slurm_time'] = args.time                # maximum time in minutes
else:
    kwargs['timeout_min'] = args.timeout_min        # minimum time in minutes

# Initialize Submitter.
executor = submitit.AutoExecutor(folder=args.slurmdir, slurm_max_num_timeout=30)
executor.update_parameters(
    name=jobname,
    slurm_partition=args.partition,
    nodes=args.nodes,
    gpus_per_node=args.gpus_per_node,
    tasks_per_node=args.gpus_per_node,  # one task per GPU
    cpus_per_task=args.cpus_per_task,
    slurm_signal_delay_s=args.signal_delay_s,
    **kwargs
)

# Create temp directory with a "snapshot" of current codebase for requeuing interrupted jobs
snapshot_dir = create_temp_dir()
print("Snapshot dir is: ", snapshot_dir)

# Select functions for run script
if args.run_script == "main.py":
    tr_function = train_function
    get_args = get_arguments
elif args.run_script == "main_pretrained.py":
    tr_function = train_function_pretrained
    get_args = get_arguments_pretrained
else:
    raise ValueError(f"Unknown run script {args.run_script}")

# Submit jobs
jobs = []
with RsyncSnapshot(snapshot_dir=snapshot_dir):
    with tqdm.tqdm(total=n_jobs) as progress_bar:
        with executor.batch():
            for cmd in cmds:
                # Use the training-file parser to parse the arguments of the command string
                parser_train = argparse.ArgumentParser('Training script', parents=[get_args()])
                job_args = parser_train.parse_args(shlex.split(cmd)[2:])    # [2:] ignores "python" and "main.py" args
                dist_url = get_init_file(job_args.exp_dir).as_uri()

                if args.is_multi_gpu_job:   # multi-gpu jobs
                    # Get the default arguments and set dist_url (main() requires these)
                    args_train_defaults = {k: parser_train.get_default(k) for k in vars(job_args).keys()}
                    job_args.defaults = args_train_defaults
                    job_args.dist_url = dist_url

                    # Submit job
                    trainer = Trainer(job_args, tr_function)

                    job = executor.submit(trainer)
                else:
                    # Submit job as a command string directly
                    cmd_as_list = re.split(r"\s+", cmd) + ["--dist-url", str(dist_url)]
                    func = submitit.helpers.CommandFunction(cmd_as_list, verbose=True)
                    job = executor.submit(func)

                jobs.append(job)
                progress_bar.update(1)

print("Finished scheduling!")
