"""Distributed helpers for PyTorch.

Borrowed from https://github.com/facebookresearch/vicreg/blob/main/distributed.py
"""


import torch
import os
import torch.distributed as dist
from pathlib import Path
import uuid
import submitit


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_init_file(exp_dir):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(exp_dir, exist_ok=True)
    init_file = os.path.join(exp_dir, f"{uuid.uuid4().hex}_init")
    if os.path.exists(init_file):
        os.remove(init_file)
    return Path(init_file)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class Trainer(object):
    def __init__(self, args, train_function):
        self.args = args
        self.train_function = train_function

    def __call__(self):
        self._setup_gpu_args()
        self.train_function(self.args)

    def checkpoint(self):
        self.args.dist_url = get_init_file(self.args.exp_dir).as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args, self.train_function)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        job_env = submitit.JobEnvironment()
        # self.args.exp_dir = os.path.join(self.args.exp_dir, job_env.job_id)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
