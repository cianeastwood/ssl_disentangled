"""General utility functions for training and evaluation."""

import argparse
import yaml
from logging import getLogger
import pickle
import os
import string
import math
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import AdamW, SGD
import torch.distributed as dist
import matplotlib.pyplot as plt

from logger import create_logger, PD_Stats

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

ALG2MLP = {
    "vicreg": "8192-8192-8192",
    "barlowtwins":  "8192-8192-8192",
    "simclr": "2048-128",
}

SHORT_AUGM_NAMES = {
    "color": "c", 
    "rotation": "r", 
    "scale": "s", 
    "transl": "t",
    "translation": "t", 
    "weak": "w",
    "medium": "m",
    "strong": "s", 
    "strongest": "ss", 
}

logger = getLogger()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    # if args.is_slurm_job:
    #     args.rank = int(os.environ["SLURM_PROCID"])
    #     args.world_size = int(os.environ["SLURM_NNODES"]) * int(
    #         os.environ["SLURM_TASKS_PER_NODE"][0]
    #     )
    # else:
    #     # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    #     # read environment variables
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, "checkpoints")
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"), rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")
    return logger, training_stats


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path,
        map_location="cuda:"
        + str(torch.distributed.get_rank() % torch.cuda.device_count()),
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(current_step, max_steps, warmup_steps, base_lr, end_lr_ratio=0.001):
    if current_step < warmup_steps:
        return base_lr * current_step / warmup_steps
    else:
        return cosine_decay(
            step=current_step - warmup_steps, 
            max_steps=max_steps - warmup_steps, 
            start_lr=base_lr, 
            end_lr=base_lr * end_lr_ratio,
            )

def cosine_decay(step, max_steps, start_lr, end_lr):
    assert step <= max_steps and start_lr > end_lr
    cosine_decay_value = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return (start_lr - end_lr) * cosine_decay_value + end_lr


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


def valid_str(v):
    if hasattr(v, '__name__'):
        return valid_str(v.__name__)
    if isinstance(v, tuple) or isinstance(v, list):
        return '-'.join([valid_str(x) for x in v])
    str_v = str(v).lower()
    valid_chars = "._%s%s" % (string.ascii_letters, string.digits)
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v


def get_exp_name(args, defaults={}, exclude_list=[], include_list=[], separator="-"):
    exp_name = ''
    args_d = args if isinstance(args, dict) else vars(args)
    for k, v in args_d.items():
        if k not in defaults:
            continue  # ignore -- added to args for other purposes, not hyperparam of interest
        if k in include_list or (v != defaults[k] and k not in exclude_list):
            if isinstance(v, bool):
                exp_name += (separator + k) if v else ''
            else:
                exp_name += separator + k + '=' + valid_str(v).replace("-", "_")   # hack replace arg separator "-"
    return exp_name.lstrip(separator)


def unnorm_imgs(imgs, mu, sigma):
    mu, sigma = np.array(mu), np.array(sigma)

    # For broadcasting, trailing dimensions must match (= num channels (C) = 3)
    imgs = imgs.transpose([1, 0, 2, 3]).T                 # (B, C, H, W) --> (W, H, B, C)
    imgs = imgs * sigma + mu
    imgs = imgs.T.transpose([1, 0, 2, 3])                 # (W, H, B, C) --> (B, C, H, W)

    return imgs


def square_grid(data):
    grid_size = math.ceil(math.sqrt(data.shape[0]))
    return grid_size, data[:grid_size ** 2]


def replace_all_occurances_in_str(s, d):
    for k, v in d.items():
        s = s.replace(k, v)
    return s

def save_image_grid(imgs, output_dir="./", title=None, max_n=25, img_mn=None, img_std=None):
    imgs = np.array(imgs[:max_n])           # plot up to sqrt(max_n) x sqrt(max_n) grid
    # Set up grid-based figure
    grid_size, imgs = square_grid(imgs)
    plt.figure(figsize=(grid_size * 2, grid_size * 2))

    # Prep images
    if img_mn is not None and img_mn is not None:
        imgs = unnorm_imgs(imgs, img_mn, img_std)
    imgs = imgs.transpose([0, 2, 3, 1])                     # (B, C, H, W) --> (B, H, W, C)
    imgs = imgs.clip(0, 1)                                  # clip RGB values to lie in [0, 1]

    # plot
    for i, img in enumerate(imgs):
        plt.subplot(grid_size, grid_size, i + 1)
        if imgs.shape[-1] == 1:  # grayscale image
            plt.imshow(img, interpolation='nearest', cmap='Greys_r')
        else:
            plt.imshow(img, interpolation='none')
        plt.axis('off')

    if title is not None:
        title = replace_all_occurances_in_str(title, SHORT_AUGM_NAMES)
        plt.suptitle(title, fontsize=18)
    else:
        title = "my_fig"

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()


def extract_interleaved_views(x, n_key_views):
    """
    Extract interleaved views from a batch of images provided by the "efficient" multi-view data-loader.
    """
    batch_size = x.shape[0]
    x_dims = x.shape[1:]
    if batch_size % n_key_views == 0:
        # Same number of key views, use tensor slicing for efficiency
        return x.view(-1, n_key_views, *x_dims).transpose(1, 0)
    else:
        # Different number of key views, use for-loop and lists 
        views = []
        for i in range(n_key_views):
            views.append(x[i::n_key_views])
        return views

def view_samples(args, xs, xs_mean, xs_std, aug_style_groups, d_path, title):
    if args.eff_mv_dl:
        # Efficient, multi-view data-loader. Views are interleaved in a batch.   
        n_key_views = args.n_views - 1
        key_view_title = "_".join(["all"] + [asg[:3] for asg in aug_style_groups])
        x_qs = extract_interleaved_views(xs[0], n_key_views)
        x_ks = extract_interleaved_views(xs[1], n_key_views) 

        if isinstance(x_qs, list):
            # Different number of key views, use for-loop to save different-sized lists
            for i, (x_q, x_k) in enumerate(zip(x_qs, x_ks)):
                save_image_grid(x_q, d_path, title + f"_q_{i}_eff", 
                                img_mn=xs_mean, img_std=xs_std)
                save_image_grid(x_k, d_path, title + f"_{key_view_title}_{i}_eff", 
                                img_mn=xs_mean, img_std=xs_std)
        else:
            # Same number of key views, use tensor slicing and save as combined grid
            x_qs = x_qs.reshape(-1, *x_qs.shape[2:])
            x_ks = x_ks.reshape(-1, *x_ks.shape[2:])
            save_image_grid(x_qs, d_path, title + "_q_eff", 
                            img_mn=xs_mean, img_std=xs_std)
            save_image_grid(x_ks, d_path, title + f"_{key_view_title}_eff", 
                            img_mn=xs_mean, img_std=xs_std)
    else:
        # Standard (possibly multi-view) data-loader.
        # If there are multiple views, they are already nicely-grouped in a list.
        for i, x in enumerate(xs):
            view_name = "q" if i == 0 else f"k{i - 1}"
            view_name += f"_{aug_style_groups[i - 2]}_std" if i >= 2 else "_std"
            save_image_grid(x, d_path, title + f"_{view_name}", img_mn=xs_mean, img_std=xs_std)
    
    # Print corresponding parameters, if available
        # TODO: add pretty_print_params() to utils.py that prints params in a readable grid
        # if "params" in batch_inputs:
        #     for i, d in enumerate(batch_inputs["params"]):
        #         print(f"{i}:" + "\n".join([f"{k}: {v}" for k, v in d.items()]))

def seed_all(seed=404, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Multi-GPU
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Sacrifice speed for exact reproducibility
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def combine_argparse_yaml_args(args, argparse_defaults, yaml_config_names=[], yaml_base_path="configs/imagenet/"):
    # Load .yaml configs
    yaml_defaults = {}
    for cname in yaml_config_names:
        with open(os.path.join(yaml_base_path, f"{cname}.yaml")) as f:
            yaml_defaults.update(yaml.load(f, Loader=yaml.FullLoader))

    # Overwrite args IF default or new
    for k, v in yaml_defaults.items():
        if k not in args or getattr(args, k) == argparse_defaults[k]:
            setattr(args, k, v)

    # Update argparse defaults with .yaml defaults
    argparse_defaults.update(yaml_defaults)


def build_dataloader(dataset, args, is_train=True):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        # prefetch_factor=1,
    )

    return loader, sampler


def build_optimizer(args, model, param_groups=None):
    if param_groups is None:
        param_groups = model.parameters()

    if args.optimizer == "LARS":
        optimizer = LARS(
            param_groups,
            lr=0,
            weight_decay=args.wd,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )
    elif args.optimizer.lower() == "adamw":
        optimizer = AdamW(param_groups, lr=0, weight_decay=args.wd)
    elif args.optimizer.lower() == "sgd":
        optimizer = SGD(param_groups, lr=0, weight_decay=args.wd, momentum=args.momentum)
    else:
        raise NotImplementedError(f"Optimizer '{args.optimizer}' not implemented.")
    return optimizer


def create_temp_dir():
    user = os.getlogin()
    random_dirname = str(uuid.uuid4())[:10]
    snapshot_dir = f"/checkpoint/{user}/tmp/{random_dirname}"
    os.makedirs(snapshot_dir, exist_ok=True)
    return Path(snapshot_dir)


def process_simclr_vissl_checkpoint(state_dict):
    state_dict_bb = state_dict['classy_state_dict']['base_model']['model']['trunk']
    state_dict_proj = state_dict['classy_state_dict']['base_model']['model']['heads']
    state_dict_bb = {
            key.replace("_feature_blocks.", ""): value
            for (key, value) in state_dict_bb.items()
        }
    state_dict_proj = {
            "projector." + key.replace("clf.0.", ""): value
            for (key, value) in state_dict_proj.items()
        }
    state_dict_bb.update(state_dict_proj)
    return state_dict_bb


def load_model(path, arch, use_projector=False):
    if '.pth' not in path:
        path = path + '.pth'
        if not os.path.exists(path):
            path = path + '.tar'

    state_dict = torch.load(path, map_location="cpu")

    filename = path.split("/")[-1]
    alg = filename[filename.index("alg=") + 4:filename.index("-")]
    mlp_structure = state_dict["mlp_structure"] if "mlp_structure" in state_dict else ALG2MLP[alg]
    projector_type = alg if "original" in filename else "vicreg"

    if "model" in state_dict:  # TODO: VICReg / BTs: need to replace module.projector too?
        state_dict = state_dict["model"]
        backbone_state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items() if "module.backbone." in key
        }
        projector_state_dict = {
            key.replace("module.projector.", ""): value
            for (key, value) in state_dict.items() if "module.projector." in key
        }
    else:   # SIMCLR PRETRAINED CKPT FROM VISSL LIB
        state_dict = process_simclr_vissl_checkpoint(state_dict)
        backbone_state_dict = {
            key: value
            for (key, value) in state_dict.items() if "projector" not in key
        }
        projector_state_dict = {
            key.replace("projector.", ""): value
            for (key, value) in state_dict.items() if "projector" in key
        }
        projector_state_dict = {
            f"{2 * int(key.split('.')[0])}{key[key.index('.'):]}": value
            for (key, value) in projector_state_dict.items()
        }

    backbone, embedding_size = eval(arch)(zero_init_residual=True)
    backbone.load_state_dict(backbone_state_dict, strict=True)

    if use_projector:
        projector = Projector(embedding_size, mlp_structure, projector_type=projector_type)
        projector.load_state_dict(projector_state_dict, strict=True)
        model = nn.Sequential(backbone, projector)
        features_size = int(mlp_structure.split("-")[-1])
    else:
        model = backbone
        features_size = embedding_size

    print('Checkpoint loaded!')
    return model, features_size


def Projector(embedding_size, mlp_structure, projector_type):
    mlp_spec = f"{embedding_size}-{mlp_structure}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))

    if projector_type in ["vicreg", "barlowtwins"]:
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1], bias=(projector_type in ["vicreg"])))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
    elif projector_type in ["simclr"]:
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]))
    return nn.Sequential(*layers)


def augm_str_to_dict(aug_strengths_str):
    """ String to dict. E.g. color=strong,scale=weak --> {"color": "strong", "scale": "weak"}. """
    if aug_strengths_str is None or len(aug_strengths_str) == 0:
        return {}
    else:
        return {s.split("=")[0]: s.split("=")[1] for s in aug_strengths_str.split(",")}


def get_augm_params_strength(augm_params, augm_strengths):
    """ Select desired strength of augmentation parameters. 
    Args:
        augm_params: dict of augmentation parameters of different strengths, e.g. {"color": {"strong": {...}, "medium": {...}, "weak": {...}}}.
        augm_strengths: dict of augmentation strengths, e.g. {"color": "strong", "scale": "weak"}.
    Returns:
        final_values: dict of augmentation parameters of desired strengths, e.g. {"color": {...}, "scale": {...}}.
    """
    final_values = {}
    for k in augm_params:           # k = "color", "scale", "norm", etc.
        if k == "norm":
            # No strength for normalization
            final_values[k] = augm_params[k]
        else:
            if k in augm_strengths:
                # Strength is specified for augmentation k, use it
                final_values[k] = augm_params[k][augm_strengths[k]]
            elif "enabled" in augm_params[k] and not augm_params[k]["enabled"]:
                # Augmentation k is disabled
                final_values[k] = None
            else:
                # Use default strength ("medium")
                final_values[k] = augm_params[k]["medium"]

    return final_values


def copy_partial_new_params_state_dict(pretr_state_dict, model_state_dict, model_name="model"):
    """
    Copy some of the weights of the current model into the pretrained state dict such that the shapes match.

    Allows layers whose shape has been changed in the current model to be (partially) initialized with 
    the pretrained weights (of a smaller size).

    E.g., if the final layer was changed from dimension 50 to 100, the first 50 weights of the pretrained
    model will be used to initialize the first 50 weights of the new model, and the remaining 50 weights
    will be initialized randomly.
    """
    for k in pretr_state_dict.keys():
        if k in model_state_dict:
            p_pretr = pretr_state_dict[k]
            p_model = model_state_dict[k]
            if p_pretr.shape != p_model.shape:
                if p_pretr.shape[1:] == p_model.shape[1:]:
                    # Only the first dimension is different: stack random weights of model on top of pre-trained weights
                    print(f"WARNING: *Partial* weights for {model_name} parameters {k} will be loaded.")
                    pretr_state_dict[k] = torch.vstack([p_pretr, p_model[p_pretr.shape[0]:]])
                elif p_pretr.shape[:-1] == p_model.shape[:-1]:
                    # Only the last dimension is different: stack random weights of model on top of pre-trained weights
                    print(f"WARNING: *Partial* weights for {model_name} parameters {k} will be loaded.")
                    pretr_state_dict[k] = torch.hstack([p_pretr, p_model[:, p_pretr.shape[1]:]])
                else:
                    raise ValueError(f"Shape mismatch for {k}: {p_pretr.shape} vs {p_model.shape}")
