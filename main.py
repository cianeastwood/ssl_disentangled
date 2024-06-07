"""Main file for training models from scratch."""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"

import argparse
import json
import sys
import time
import signal
import copy
# from turtle import forward

import torch
import torch.nn.functional as F
from collections import defaultdict
from torch import nn
from torch.optim import AdamW, SGD
import torchvision.datasets as datasets

from utils import adjust_learning_rate, exclude_bias_and_norm, LARS, get_exp_name
from utils import handle_sigusr1, handle_sigterm, seed_all, MultiEpochsDataLoader
from utils import save_image_grid, combine_argparse_yaml_args, build_dataloader
from utils import augm_str_to_dict, get_augm_params_strength
from algorithms import get_algorithm_class
import augmentations as aug
from datasets import TriFeatureFast, ColorDSprites, DSprites

from distributed import init_distributed_mode
from pytorch_lightning.loggers import WandbLogger

import resnet

# Cluster re-requeuing / interruptions
signal.signal(signal.SIGUSR1, handle_sigusr1)
signal.signal(signal.SIGTERM, handle_sigterm)

EPS = 1e-6


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Algorithm / method
    parser.add_argument("--alg", default="simclr",
                        help='Algorithm name', choices=["simclr", "barlowtwins", "vicreg", "simclrmultihead", 
                                                        "vicregmultihead", "simclrlooc", "simclraugself", "vicregaugself"])

    # Data
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=["imagenet", "imagenet_blurred", "cifar10", "cifar100", "colordsprites", "trifeature"],
                        help='Dataset name')
    parser.add_argument("--data-dir", type=str,
                        help='Path to the image net dataset')
    parser.add_argument("--aug-type", type=str,
                        help='Type of augmentation to use: symmetric or asymmetric.')
    parser.add_argument("--aug-strengths", type=str,
                        help='Specify stronger or weaker augmentations, e.g. color=strong,scale=weak.')
    parser.add_argument("--aug-style-groups", type=str,
                        help='Specify additional embedding spaces for "style" variables via augmentation groups, \
                              e.g., "appearance,spatial".')
    parser.add_argument("--view-samples", action="store_true",
                        help='Saved some augmented samples as they would be fed into the model.')
    parser.add_argument("--n-colors-dsprites", type=int, default=10,
                        help='Number of color categories for the colorDsprites dataset in [1, 10].')

    # Checkpoints
    parser.add_argument("--exp-dir", type=str, default="/checkpoint/ceastwood/ssl_disent",
                        help='Path to the experiment folder, where all outputs will be stored')
    parser.add_argument("--log-freq", type=float, default=0.5,
                        help='Save train logs every [log-freq] epochs (float).')
    parser.add_argument("--eval-freq", type=float, default=10.,
                        help='Save eval logs every [eval-freq] epochs (float).')
    parser.add_argument("--ckpt-freq", type=int, default=20,
                        help='Save checkpoint every [ckpt-freq] epochs (int).')
    parser.add_argument("--pretr-ckpt-pth", type=str,
                        help='Path to checkpoint to resume from.')
    parser.add_argument("--ckpt-save-epochs-sep", type=str,     #     [1, 10, 20, 50, 100]
                        help='Save *separate* checkpoints (i.e., different paths) at these comma-separated epochs, e.g.: 1, 10, 50.')

    # Model / arch
    parser.add_argument("--arch", type=str,
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp",
                        help='Size and number of layers of the MLP expander head')  # default in config files
    parser.add_argument("--adapt-resnet", action="store_true",
                        help='Adapt resnet filter sizes for smaller images (e.g. 64x64)')
    parser.add_argument("--embedding-split-fracts", type=str,
                        help='Specify the fraction of the embedding space to use for each group, e.g. "0.5,0.25,0.25".')

    # Optim
    parser.add_argument("--optimizer", type=str, default='LARS', choices=["LARS", "adamw", "sgd"])
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256 \
                              Defaults in algorithm config files.')
    parser.add_argument("--warm-up-epochs", type=int, default=10,
                        help='Number of warm-up epochs')
    parser.add_argument("--end-lr-ratio", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='Momentum of SGD solver')
    parser.add_argument("--seed", type=int, default=1234)

    # Algorithms / loss terms
    # -- VICReg
    parser.add_argument("--sim-coeff", type=float,
                        help='Invariance regularization loss coefficient for VICReg.')
    parser.add_argument("--std-coeff", type=float,
                        help='Variance regularization loss coefficient for VICReg.')
    parser.add_argument("--cov-coeff", type=float,
                        help='Covariance regularization loss coefficient for VICReg.')
    # -- SimCLR
    parser.add_argument("--temperature", type=float,
                        help='SimCLR temperature.')
    # -- BTS
    parser.add_argument("--lambd-bts", type=float,
                        help='Weighting of the negative/off-diag term in BarlowTwins.')
    # -- OURS
    parser.add_argument("--lambd", type=float,
                        help='(Initial) weighting of the invariance constraint. Dataset-dependent default in config files.')
    parser.add_argument('--base-lr-lambda', type=float, default=0.,
                        help='Learning rate or step size for adapting lambda. Default=0 means no adaptation.')
    parser.add_argument('--tolerance', type=float, default=0.,
                        help='Constraint tolerance for the positive term value.')
    parser.add_argument('--exclude-fract', type=float, default=0.,
                        help='Exclude the alpha-largest positive terms (largest invariance terms for samples).')
    parser.add_argument("--warm-up-epochs-lambda", type=int,
                        help='Warm-up epochs before starting to adapt lambda. Dataset-dependent default in config files.')
    parser.add_argument("--inner-step-increm", type=int, default=0,
                        help='Inner-step increments per epoch.')
    parser.add_argument("--simclr-cross-key-negs", action="store_true", 
                        help="Use cross-key negatives for SimCLR, i.e., encourage key-views to be dissimiliar from each other.")

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument("--not_fp16", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Simplified runs for fast debugging.")

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="Use wandb logging")
    parser.add_argument("--entity", type=str, default="ceastwood",
                        help="Wandb username")
    parser.add_argument("--project", type=str, default="ssl_disent",
                        help="Wandb project name.")
    parser.add_argument("--sweep-name", type=str, default="test_logging",
                        help="Wandb sweep/group name (optional).")

    # YAML files
    parser.add_argument("--use-yaml-configs", action="store_true",
                        help="Overwrite the argsparse defaults with the defaults store in .yaml config files.")

    return parser


def preprocess_args(args):
    # Load .yaml configs and overwrite iff arg == argparse_default
    # TODO: remove hardcoding, and use the args.use_yaml_configs flag
    config_dataset = args.dataset.replace("_blurred", "")               # imagenet and imagenet_blurred share the same config
    base_path = f"configs/{config_dataset}/"
    yaml_config_names = ["dataset", args.alg]
    combine_argparse_yaml_args(args, args.defaults, yaml_config_names, base_path)

    # Get experiment name based on non-default args
    exlcude_list = ['arch', 'log_freq', 'eval_freq', 'num_workers', "exp_dir", "data_dir",
                    "wandb", "project", "entity", "sweep_name", "use_yaml_configs", "local_rank",
                    "view_samples", "dist_url", "defaults", "world_size", "aug_params",
                    "ckpt_save_epochs_sep", "ckpt_freq", "aug_style_groups", "embedding_split_fracts",]
    include_list = ['alg', 'dataset']
    args.exp_name = get_exp_name(args, args.defaults, exlcude_list, include_list)


def main(args):
    # Setup
    torch.backends.cudnn.benchmark = True
    preprocess_args(args)
    init_distributed_mode(args)
    gpu = torch.device(args.device)     # actual device
    assert args.batch_size % args.world_size == 0
    seed_all(args.seed)

    # Logging
    print("Setting up:")
    print("\t logging...")

    args.ckpt_dir = os.path.join(args.exp_dir, "ckpt")
    args.logs_dir = os.path.join(args.exp_dir, "logs")
    args.exp_ckpt_pth = os.path.join(args.ckpt_dir, args.exp_name + ".pth")
    if args.ckpt_save_epochs_sep is not None:
        ckpt_save_epochs_sep = [int(e) for e in args.ckpt_save_epochs_sep.split(",")]
    else:
        ckpt_save_epochs_sep = []
    print(args.exp_ckpt_pth)
    print(ckpt_save_epochs_sep)

    if args.rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)

        if args.wandb:
            import wandb
            wandb_logger = WandbLogger(
                name=args.exp_name,
                save_dir=args.logs_dir,
                entity=args.entity,
                project=args.project,
                group=args.sweep_name,
                settings=wandb.Settings(start_method="fork"),
                resume="allow",
            )
            wandb_logger.log_hyperparams(args)
        else:
            stats_file = open(os.path.join(args.logs_dir,  args.exp_name + ".txt"), "a", buffering=1)
            print(" ".join(sys.argv))
            print(" ".join(sys.argv), file=stats_file)

    # Data          TODO: create and use validation set for each dataset (rather than test set)
    print("\t dataset, transforms and loaders...")
    
    # -- Prep augmentation parameters
    aug_params = copy.deepcopy(args.aug_params)
    aug_params.update(aug_params[args.aug_type])           # add symmetric- or asymmetric-specific params directly into dict
    for a_type in ["symmetric", "asymmetric"]:
        aug_params.pop(a_type)                                    # remove symmetric- or asymmetric-specific params sub-dict
    aug_strength_dict = augm_str_to_dict(args.aug_strengths)
    aug_params = get_augm_params_strength(aug_params, aug_strength_dict)

    # -- Prep multi-head setup
    aug_style_groups = args.aug_style_groups.split(",") if args.aug_style_groups is not None else ()
    args.z_split_fracts = [float(f) for f in args.embedding_split_fracts.split(",")] if args.embedding_split_fracts is not None else ()
    args.n_views = 2 + len(aug_style_groups)
    if len(args.z_split_fracts) > 0:    # multi-head
        assert len(args.z_split_fracts) == len(aug_style_groups) + 1, "Must specify an embedding-split fraction for each augmentation group."
        assert ("multihead" in args.alg) or ("looc" in args.alg), "Must specify multi-head algorithm for multi-head training."
        print(f"Warning: multi-head training with {args.n_views} views per example (one per head), rather than the usual 2 views per example. " + \
              f"Total batch size across the {args.world_size} GPUs is thus {args.batch_size * args.n_views} instead of {args.batch_size * 2}, " + \
              f"with {(args.batch_size // args.world_size) * args.n_views} images per GPU instead of {(args.batch_size // args.world_size) * 2}. " + \
                "Ensure that the specified batch size is adjusted accordingly.")
    else:
        if ("multihead" in args.alg) or ("looc" in args.alg):
            raise ValueError(f"Multi-head algorithm specified---{args.alg}---but not" + \
                             f"multi-head training (via embedding splits---{args.z_split_fracts}).")

    # -- Build transformations and datasets
    train_transforms = aug.TrainTransformMultiViewParams(args.dataset, args.image_shape[-1], args.aug_type, 
                                                         aug_params, aug_style_groups)
    val_transforms = aug.ValTransform(args.dataset, args.image_shape[-1], aug_params)
    aug_type = f"{args.aug_type}_{args.aug_strengths}" if args.aug_strengths is not None else args.aug_type
    
    if args.dataset == "imagenet":
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), val_transforms)
    elif args.dataset == "imagenet_blurred":        # args.data_dir_blurred so that the configs are shared with imagenet
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir_blurred, "train_blurred"), train_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir_blurred, "val_blurred"), val_transforms)
    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=train_transforms)
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=val_transforms)
    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(args.data_dir, train=True, transform=train_transforms)
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=val_transforms)
    elif args.dataset == "dsprites":
        train_dataset = DSprites(args.data_dir, split="train", transform=train_transforms)
        val_dataset = DSprites(args.data_dir, split="test", transform=val_transforms)
    elif args.dataset == "colordsprites":
        train_dataset = ColorDSprites(args.data_dir, split="train", transform=train_transforms, 
                                      n_colors=args.n_colors_dsprites, debug=args.debug)
        val_dataset = ColorDSprites(args.data_dir, split="test", transform=val_transforms,
                                    n_colors=args.n_colors_dsprites, debug=args.debug)
    elif args.dataset == "trifeature":
        train_dataset = TriFeatureFast(args.data_dir, split="train", transform=train_transforms,
                                       image_size=args.image_shape[-1])
        val_dataset = TriFeatureFast(args.data_dir, split="test", transform=val_transforms,
                                     image_size=args.image_shape[-1])
    else:
        raise ValueError(f"Invalid dataset {args.dataset}.")

    train_loader, train_sampler = build_dataloader(train_dataset, args, is_train=True)
    val_loader, _ = build_dataloader(val_dataset, args, is_train=False)

    if args.view_samples and args.rank == 0:
        # Setup
        print("\t image-grid of samples...")
        d_path = "samples/"         # os.path.join(args.exp_dir, "samples")
        os.makedirs(d_path, exist_ok=True)
        batch_inputs = make_inputs(next(iter(train_loader)))
        title = args.dataset + "_" + aug_type
        
        # Save grid of images
        for i, x in enumerate(batch_inputs["views"]):
            view_name = "q" if i == 0 else f"k{i - 1}"
            view_name += f"_{aug_style_groups[i - 2]}" if i >= 2 else ""
            save_image_grid(x, d_path, title + f"_{view_name}", 
                            img_mn=train_transforms.mean, img_std=train_transforms.std)
        
        # Print corresponding parameters, if available
        # TODO: add pretty_print_params() to utils.py that prints params in a readable grid
        # if "params" in batch_inputs:
        #     for i, d in enumerate(batch_inputs["params"]):
        #         print(f"{i}:" + "\n".join([f"{k}: {v}" for k, v in d.items()]))

    # Arch (h = embedding = post-backbone, pre-projection)
    print("\t model / arch...")
    backbone, args.h_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
    if args.adapt_resnet and args.image_shape[-1] < 100:
        resnet.adapt_resnet(backbone, args.arch, args.image_shape)
    projector = Projector(args.h_size, args.mlp)

    # Algorithm
    alg_class = get_algorithm_class(args.alg)
    model = alg_class(args, backbone, projector).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Optimizer
    if args.optimizer == "LARS":
        optimizer = LARS(
            model.parameters(),
            lr=0,
            weight_decay=args.wd,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )
    elif args.optimizer.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=0, weight_decay=args.wd)
    elif args.optimizer.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=0, weight_decay=args.wd, momentum=args.momentum)
    else:
        raise NotImplementedError(f"Optimizer '{args.optimizer}' not implemented.")
    
    # Load checkpoint, if available
    def load_ckpt(ckpt_pth):
        if args.rank == 0:
            print(f"Resuming from checkpoint {ckpt_pth}")
        ckpt = torch.load(ckpt_pth, map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt["epoch"]

    if os.path.isfile(args.exp_ckpt_pth):
        # Resume from checkpoint path with same name as current experiment
        start_epoch = load_ckpt(args.exp_ckpt_pth)
    else:
        if args.pretr_ckpt_pth is not None:
            # Start with a specified pretrained model, e.g. best/original SimCLR model
            start_epoch = load_ckpt(args.pretr_ckpt_pth)
        else:
            # Start from scratch (random init)
            start_epoch = 0

    # Learning rate scheduling
    max_steps = args.epochs * len(train_loader)
    warmup_steps = args.warm_up_epochs * len(train_loader)
    base_lr = args.base_lr * args.batch_size / 256.
    warmup_steps_lambda = args.warm_up_epochs_lambda * len(train_loader)
    lr_lambda = base_lr_lambda = args.base_lr_lambda * args.batch_size / 256.
    
    # Final setup
    n_inner_steps = 1
    lambd = args.lambd * torch.ones(1 + len(aug_style_groups), requires_grad=False).cuda(gpu)
    inv_constr_keys = [f"inv_constraint_{i}" for i in range(len(lambd))] if len(lambd) > 1 else ["inv_constraint"]
    if args.debug:
        args.ckpt_freq = 1
        log_freq_step = eval_freq_step = 1
        n_val_batches = 1
    else:
        log_freq_step = int(len(train_loader) * args.log_freq)      # allows args.log_freq < 1 for debugging
        eval_freq_step = int(len(train_loader) * args.eval_freq)    # allows args.eval_freq < 1 for debugging
        n_val_batches = len(val_loader)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    print("Setup complete. Starting to train...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for step, input_batch in enumerate(train_loader, start=epoch * len(train_loader)):
            # Forward pass
            if args.not_fp16:
                loss, logs = model.forward(make_inputs(input_batch, gpu), lambd=lambd)
            else:
                with torch.cuda.amp.autocast():
                    loss, logs = model.forward(make_inputs(input_batch, gpu), lambd=lambd)
            
            # Outer/dual step (\lambda) every n_inner_steps after warmup_steps_lambda if adaptive (base_lr_lambda > 0)
            if base_lr_lambda > 0 and step >= warmup_steps_lambda and step % int(n_inner_steps) == 0:
                with torch.no_grad():
                    lr_lambda = adjust_learning_rate(step, max_steps, warmup_steps_lambda, base_lr_lambda, args.end_lr_ratio)
                    constraint_values = [model.module.inv_constraint_loss(logs[k].item(), space=i) 
                                         for i, k in enumerate(inv_constr_keys)]
                    lambd = update_lambdas(lambd, constraint_values, lr_lambda, args.tolerance)

            # Inner/primal step (\theta)
            lr = adjust_learning_rate(step, max_steps, warmup_steps, base_lr, args.end_lr_ratio)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.zero_grad()
            if args.not_fp16:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Logging
            log_train = (step + 1) % log_freq_step == 0
            log_eval = (step + 1) % eval_freq_step == 0

            if log_train:  # aggregate train logs
                for v in logs.values():
                    torch.distributed.reduce(v.div_(args.world_size), 0)

            if log_eval:   # aggregate (cumulative) eval logs
                cum_eval_logs = evaluate(model, val_loader, args, gpu)

            if args.rank == 0 and (log_train or log_eval):
                # Add train and eval logs
                logs_ = {}
                if log_train:
                    logs_.update({k: v.item() for k, v in logs.items()})
                    if len(lambd) > 1:
                        logs_.update({f"lambda_{i}": l.item() for i, l in enumerate(lambd)})
                    else:
                        logs_.update({"lambda": lambd.item()})
                if log_eval:
                    logs_.update({k + "_val": v / n_val_batches for k, v in cum_eval_logs.items()})

                # Sort, add base/general logs and round
                logs_ = dict(sorted(logs_.items()))
                all_logs = dict(epoch=epoch, step=step, time=int(time.time() - start_time), lr=lr, lr_lambda=lr_lambda)
                all_logs.update(logs_)      # add afterwards to allow "epoch" first but logs_ to be sorted
                all_logs = {k: round(v, 6) for k, v in all_logs.items()}

                if args.wandb:
                    wandb_logger.log_metrics(all_logs, step=step)
                else:
                    print(json.dumps(all_logs), file=stats_file)

                print(json.dumps(all_logs))

            if args.debug:
                break
        
        # Increment the number of inner/primal steps per outer/dual step
        n_inner_steps += args.inner_step_increm
        
        # Save checkpoints at different epochs for subsuequent training
        if args.rank == 0 and ((epoch + 1) in ckpt_save_epochs_sep):
            save_ckpt(model, optimizer, args, epoch, args.exp_ckpt_pth.replace(".pth", f"-e={epoch + 1}.pth"))
        
        # Save checkpoints at regular intervals, overwriting previous ones
        print()
        if args.rank == 0 and (((epoch + 1) == args.epochs) or ((epoch + 1) % args.ckpt_freq == 0)):
            save_ckpt(model, optimizer, args, epoch, args.exp_ckpt_pth)
        
        if args.debug:
            break
    print("Training complete.")


def evaluate(model, val_loader, args, gpu):
    model.eval()
    cumulative_logs = defaultdict(float)
    for input_batch in val_loader:
        with torch.no_grad():
            if args.not_fp16:
                _, logs = model.forward(make_inputs(input_batch, gpu), is_val=True)
            else:
                with torch.cuda.amp.autocast():
                    _, logs = model.forward(make_inputs(input_batch, gpu), is_val=True)

        # aggregate logs
        for v in logs.values():
            torch.distributed.reduce(v.div_(args.world_size), 0)

        # cumulative values
        for key, value in logs.items():
            cumulative_logs[key] += value.item()
        
        if args.debug:
            break

    model.train()

    return cumulative_logs


def save_ckpt(model, optimizer, args, epoch, save_pth):
    state = dict(
        epoch=epoch + 1,
        model=model.module.state_dict(),            # go inside nn.DataParallel to get the module
        optimizer=optimizer.state_dict(),
        mlp_structure=args.mlp,
        z_split_fracts=args.z_split_fracts,
    )
    torch.save(state, save_pth)


def make_inputs(batch_samples, gpu=None):
    if isinstance(batch_samples[0][0], list):
        (views, params), labels = batch_samples
        if gpu is None:
            return dict(views=views, params=params, labels=labels)
        return dict(
            views=[view.cuda(gpu, non_blocking=True) for view in views],
            params=[params_dict_to_gpu(params_dict, gpu) for params_dict in params],
            labels=labels.cuda(gpu, non_blocking=True),
        )
    views, labels = batch_samples
    if gpu is None:
        return dict(views=views, labels=labels)
    return dict(
        views=[view.cuda(gpu, non_blocking=True) for view in views],
        labels=labels.cuda(gpu, non_blocking=True),
    )


def params_dict_to_gpu(params_dict, gpu):
    for k, v in params_dict.items():
        if isinstance(v, torch.Tensor):
            params_dict[k] = v.cuda(gpu, non_blocking=True)
        elif isinstance(v, list):
            params_dict[k] = [v_.cuda(gpu, non_blocking=True) for v_ in v]  # list of tensors, possible of different types
        else:
            params_dict[k] = v          # no gpu for this param (uknown type)
    return params_dict


def update_lambdas(lambdas, constraint_values, lr, tolerance):
    lambdas_out = torch.empty_like(lambdas)         # avoid inplace ops
    for i in range(len(lambdas)):
        constraint_unsatisfied_i = constraint_values[i] - tolerance
        lambdas_out[i] = F.relu(lambdas[i] + lr * constraint_unsatisfied_i)
    return lambdas_out

def Projector(embedding_size, mlp_structure_str):
    mlp_spec = f"{embedding_size}-{mlp_structure_str}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training script', parents=[get_arguments()])
    args = parser.parse_args()
    args.defaults = {k: parser.get_default(k) for k in vars(args).keys()}
    main(args)
