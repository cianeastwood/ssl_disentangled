"""Main file for fine-tuning a pre-trained model."""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"

import math
from pathlib import Path
import argparse
import json
import sys
import time
import signal
import yaml
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
from utils import augm_str_to_dict, get_augm_params_strength, build_optimizer
from utils import copy_partial_new_params_state_dict, view_samples
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

    # Additional arguments for starting with a pretrained model
    # -- general
    parser.add_argument("--proj-only-tuning", default="all",
                        help='Type of projector (fine)tuning before updating the whole model', 
                        choices=["content-linear-all", "linear-all", "all", "all-scratch", "none", "none-scratch"])
    # -- epochs
    parser.add_argument("--linear-only-epochs", type=int, default=20,
                        help='Epochs to train the last layer of the projector only')
    parser.add_argument("--linear-only-warm-up-epochs", type=int, default=2,
                        help='Epochs to warm up when training the projector only')
    parser.add_argument("--proj-only-epochs", type=int, default=50,
                        help='Epochs to train the projector only')
    parser.add_argument("--proj-only-warm-up-epochs", type=int, default=5,
                        help='Epochs to warm up when training the projector only')
    # -- lrs
    parser.add_argument("--base-lr-proj", type=float,
                        help='Base learning rate for the projector when everything is being trained (second phase). \
                        Effective learning after warmup is [base-lr] * [batch-size] / 256. Defaults in algorithm config files.')
    parser.add_argument("--base-lr-bb", type=float,
                        help='Base learning rate for the backbone when everything is being trained (second phase). \
                        Effective learning after warmup is [base-lr] * [batch-size] / 256. Defaults in algorithm config files.')
    parser.add_argument("--base-lr-proj-only", type=float,
                        help='Base learning rate when tuning only the projector in an initial phase. Effective learning after warmup is \
                        [base-lr] * [batch-size] / 256. Defaults in algorithm config files.')
    parser.add_argument("--proj-only-end-lr-ratio", type=float, default=0.01,
                        help="Less aggressive learning rate decay for projector-only (quick, rough estimate is the goal).")
    
    
    # Algorithm / method
    parser.add_argument("--alg", default="simclr",
                        help='Algorithm name', 
                        choices=["simclr", "barlowtwins", "vicreg", "simclrmultihead", "vicregmultihead", 
                                 "simclrlooc", "simclraugself", "vicregaugself", "simclrmultiheadefficient",
                                 "vicregmultiheadefficient"])

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
    parser.add_argument("--eff-mv-dl", action="store_true", help="Use efficient multi-view dataloader.")
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
    parser.add_argument("--mlp", type=str,
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
    parser.add_argument("--m-cont-views", action="store_true", 
                        help="Use multiple positive views for the content space, i.e., the keys of other spaces too.")
    parser.add_argument("--adapt-cont-lambd", action="store_true",
                        help="Adapt the positive-weighting term for content space.")
    parser.add_argument("--constr-crop", action="store_true",
                        help="Use a 'contrained' crop for style-space keys that has a smaller/weaker difference with the query.")
    parser.add_argument("--add-cont-neg", action="store_true", 
                        help="Use additional negatives for the content space.")
    parser.add_argument("--mlp-style", type=str,
                        help='Size and number of layers of the style-MLP expander head')  # default in config files

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
                    "ckpt_save_epochs_sep", "ckpt_freq", "aug_style_groups", "embedding_split_fracts",
                    "pretr_ckpt_pth", "batch_size", "eff_mv_dl", "add_cont_neg", "constr_crop",]
    include_list = ['alg', 'dataset', 'proj_only_tuning']
    args.exp_name = get_exp_name(args, args.defaults, exlcude_list, include_list)
    if "-e=" in args.pretr_ckpt_pth:
        e = args.pretr_ckpt_pth.split("-e=")[-1].split(".")[0]
        args.pretr_epochs_str = f"-e={e}"
    else:
        args.pretr_epochs_str = ""
    args.exp_name += args.pretr_epochs_str

def main(args):
    # Setup
    overall_start_time = time.time()
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
    print(args.exp_ckpt_pth)

    wandb_logger, stats_file = None, None
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

    # -- Prep multi-head setup and do some checks (TODO: move this somewhere else, e.g., utils.py)
    aug_style_groups = args.aug_style_groups.split(",") if args.aug_style_groups is not None else ()
    args.z_split_fracts = [float(f) for f in args.embedding_split_fracts.split(",")] if args.embedding_split_fracts is not None else ()
    args.n_views = 2 + len(aug_style_groups)
    if len(args.z_split_fracts) > 0:    # multi-head training
        assert len(args.z_split_fracts) == len(aug_style_groups) + 1, "Must specify an embedding-split fraction for each augmentation group."
        assert ("multihead" in args.alg) or ("looc" in args.alg), "Must specify multi-head algorithm for multi-head training."
        if args.eff_mv_dl:
            if args.alg not in ["simclrmultihead", "vicregmultihead"]:
                raise ValueError(f"Efficient dataloading specified but not an appropriate 'efficient' algorithm---{args.alg}---that" + \
                                 f"supports it.")
        else:
            print(f"Warning: multi-head training with {args.n_views} views per example (one per head), rather than the usual 2 views per example. " + \
                f"Total batch size across the {args.world_size} GPUs is thus {args.batch_size * args.n_views} instead of {args.batch_size * 2}, " + \
                f"with {(args.batch_size // args.world_size) * args.n_views} images per GPU instead of {(args.batch_size // args.world_size) * 2}. " + \
                    "Ensure that the specified batch size is adjusted accordingly.")
    else:
        if ("multihead" in args.alg) or ("looc" in args.alg):
            raise ValueError(f"Multi-head algorithm specified---{args.alg}---but not" + \
                             f"multi-head training (via embedding splits---{args.z_split_fracts}).")

    # -- Build transformations and datasets
    if args.eff_mv_dl:
        train_transforms = aug.TrainTransformMultiViewParamsEfficient(args.dataset, args.image_shape[-1], 
                                                                      args.aug_type, aug_params, aug_style_groups,
                                                                      n_workers=args.num_workers, constr_crop=args.constr_crop)
    else:
        train_transforms = aug.TrainTransformMultiViewParams(args.dataset, args.image_shape[-1], 
                                                             args.aug_type, aug_params, aug_style_groups,
                                                             constr_crop=args.constr_crop)
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
        print("\t image-grid of samples...")
        d_path = "samples/"         # os.path.join(args.exp_dir, "samples")
        os.makedirs(d_path, exist_ok=True)
        batch_inputs = make_inputs(next(iter(train_loader)))
        title = args.dataset + "_" + aug_type
        xs = batch_inputs["views"]
        view_samples(args, xs, train_transforms.mean, train_transforms.std, aug_style_groups, d_path, title)
    
    # Arch (h = embedding = post-backbone, pre-projection)
    print("\t model / arch...")
    backbone, args.h_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
    if args.adapt_resnet and args.image_shape[-1] < 100:
        resnet.adapt_resnet(backbone, args.arch, args.image_shape)
    projector = Projector(args.h_size, args.mlp)
    if args.mlp_style is not None:
        projector_style = Projector(args.h_size, args.mlp_style)
    else:
        projector_style = None

    # Preprocess and load pretrained checkpoint
    if args.pretr_ckpt_pth is None or not os.path.exists(args.pretr_ckpt_pth):
        raise ValueError(f"No pretrained model found to start from: {args.pretr_ckpt_pth}.")
    state_dict = torch.load(args.pretr_ckpt_pth, map_location="cpu")["model"]
    def get_subset_state_dict(state_dict, prefix):
        return {
            key.replace(prefix, ""): value
            for (key, value) in state_dict.items() if prefix in key
        }
    # --- separate out the different components of the model
    backbone_state_dict = get_subset_state_dict(state_dict, "backbone.")
    projector_state_dict = get_subset_state_dict(state_dict, "projector.")
    classifier_state_dict = {c_name: get_subset_state_dict(state_dict, c_name + ".") 
                             for c_name in ["classifier_z", "classifier_h"]}
    # --- load the backbone weights and allow partial projector weights to be loaded (when mlp size changes)
    backbone.load_state_dict(backbone_state_dict, strict=True)
    print(f"Loaded backbone weights from {args.pretr_ckpt_pth}.")
    copy_partial_new_params_state_dict(projector_state_dict, projector.state_dict(), "projector")

    # Algorithm
    alg_class = get_algorithm_class(args.alg)
    model = alg_class(args, backbone, projector, projector_style).cuda(gpu)

    # Classifier
    # -- load pretrained weights
    for c_name, pretr_sd in classifier_state_dict.items():
        classifier = getattr(model, c_name)
        pretr_sd = {k: v.cuda(gpu) for k, v in pretr_sd.items()}
        copy_partial_new_params_state_dict(pretr_sd, classifier.state_dict(), c_name)
        classifier.load_state_dict(pretr_sd, strict=True)
        print(f"Loaded {c_name} weights from {args.pretr_ckpt_pth}.")
    # -- add param groups
    param_groups_classifier_z = []
    if projector_style is None: 
        param_groups_classifier_z.append(dict(params=model.classifier_z.parameters(), lr=0, name="classifier_z"))
    else:
        param_groups_classifier_z.append(dict(params=model.classifier_z_style.parameters(), lr=0, name="classifier_z_style"))
        param_groups_classifier_z.append(dict(params=model.classifier_z_joint.parameters(), lr=0, name="classifier_z_joint"))
    
    # Final setup
    for lr_attr in ["base_lr_bb", "base_lr_proj", "base_lr_proj_only"]:
        if getattr(args, lr_attr) is None:  # if not specified, set to base_lr
            setattr(args, lr_attr, args.base_lr)
    lambd = args.lambd * torch.ones(1 + len(aug_style_groups), requires_grad=False).cuda(gpu)
    inv_constr_keys = [f"inv_constraint_{i}" for i in range(len(lambd))] if len(lambd) > 1 else ["inv_constraint"]
    eff_start_epoch = 0
    if args.debug:
        args.ckpt_freq = 1
        log_freq_step = eval_freq_step = 1
        n_val_batches = 1
    else:
        log_freq_step = int(len(train_loader) * args.log_freq)      # allows args.log_freq < 1 for debugging
        eval_freq_step = int(len(train_loader) * args.eval_freq)    # allows args.eval_freq < 1 for debugging
        n_val_batches = len(val_loader)

    # ################################################
    # TRAIN ONLY THE PROJECTOR FIRST
    # ################################################
    for p in model.backbone.parameters():
        p.requires_grad = False
    
    if "linear" in args.proj_only_tuning:
        # Fine-tune the last layer of the projector only
        model.projector.load_state_dict(projector_state_dict, strict=True)
        print(f"Loaded projector weights from {args.pretr_ckpt_pth}.")

        # Only optimize the last layer of the projector
        final_proj_layer = list(model.projector)[-1]                              # last layer of projector
        param_groups = [dict(params=final_proj_layer.parameters(), lr=0, name="projector")]
        param_groups += param_groups_classifier_z                             # add classifier param groups

        # Train
        exp_ckpt_pth = args.exp_ckpt_pth.replace(".pth", "-tune=l.pth")     # separate checkpoint path per phase
        eff_time_ = int(time.time() - overall_start_time)
        print("Fine-tuning only the last projector layer.")
        model, lambd = train(model, train_loader, val_loader, args, gpu, train_sampler, param_groups,
                        wandb_logger, stats_file, lambd, inv_constr_keys, proj_only=True, epochs=args.linear_only_epochs, 
                        warm_up_epochs=args.linear_only_warm_up_epochs, log_freq_step=log_freq_step, 
                        eval_freq_step=eval_freq_step, n_val_batches=n_val_batches, 
                        exp_ckpt_pth=exp_ckpt_pth, eff_start_epoch=eff_start_epoch, eff_time_start=eff_time_,)
        eff_start_epoch += args.linear_only_epochs
    
    if "all" in args.proj_only_tuning:
        # Fine-tune the whole projector
        if model.projector_style is None:
            # no separate style projector, so fine-tune the previous projector
            param_groups = [dict(params=model.projector.parameters(), lr=0, name="projector")]
            for p in projector.parameters():
                p.requires_grad = True
        else:
            # new, separate style projector to learn from scratch (no pretraining) -- don't fine-tune the previous (content) projector
            param_groups = [dict(params=model.projector_style.parameters(), lr=0, name="projector_style")]
            for p in projector.parameters():
                p.requires_grad = False
        
        # More setup
        param_groups += param_groups_classifier_z                             # add classifier param groups
        if not ("scratch" in args.proj_only_tuning or "linear" in args.proj_only_tuning):
            # use pretrained projector weights, and they have not already been loaded
            model.projector.load_state_dict(projector_state_dict, strict=True)
            print(f"Loaded projector weights from {args.pretr_ckpt_pth}.")
        exp_ckpt_pth = args.exp_ckpt_pth.replace(".pth", "-tune=a.pth")     # separate checkpoint path per phase
        eff_time_ = int(time.time() - overall_start_time)
        
        # Train
        print("Fine-tuning the projector.")
        model, lambd = train(model, train_loader, val_loader, args, gpu,
                             train_sampler, param_groups, wandb_logger, stats_file, lambd, inv_constr_keys,
                             proj_only=True, epochs=args.proj_only_epochs, warm_up_epochs=args.proj_only_warm_up_epochs,
                             log_freq_step=log_freq_step, eval_freq_step=eval_freq_step, n_val_batches=n_val_batches,
                             exp_ckpt_pth=exp_ckpt_pth, eff_start_epoch=eff_start_epoch, eff_time_start=eff_time_,)
        eff_start_epoch += args.proj_only_epochs
    else:
        # No fine-tuning of the projector by itself
        if args.proj_only_tuning == "none":             # start with pretrained projector-weights
            model.projector.load_state_dict(projector_state_dict, strict=True)
            print(f"Loaded projector weights from {args.pretr_ckpt_pth}.")
        
        elif args.proj_only_tuning == "none-scratch":   # start with random projector-weights
            pass
        
        else:
            raise ValueError(f"Invalid projector tuning {args.proj_only_tuning}.")

    # ################################################
    # TRAIN WHOLE MODEL (BACKBONE + PROJECTOR)
    # ################################################

    # Grads for all params
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = True
    
    # Optimize everything (separate names for separate learning rates)
    param_groups = [dict(params=model.backbone.parameters(), lr=0, name="backbone"),
                    dict(params=model.projector.parameters(), lr=0, name="projector"),
                    dict(params=model.classifier_h.parameters(), lr=0, name="classifier_h"),
                    dict(params=model.classifier_z.parameters(), lr=0, name="classifier_z")]
    if model.projector_style is not None:
        param_groups.extend([dict(params=model.projector_style.parameters(), lr=0, name="projector_style"),
                             dict(params=model.classifier_z_style.parameters(), lr=0, name="classifier_z_style"),
                             dict(params=model.classifier_z_joint.parameters(), lr=0, name="classifier_z_joint")])
    
    # Train
    eff_time_ = int(time.time() - overall_start_time)
    print("Fine-tuning the entire model (backbone + projector).")
    train(model, train_loader, val_loader, args, gpu,
          train_sampler, param_groups, wandb_logger, stats_file, lambd, inv_constr_keys,
          proj_only=False, epochs=args.epochs, warm_up_epochs=args.warm_up_epochs,
          log_freq_step=log_freq_step, eval_freq_step=eval_freq_step, n_val_batches=n_val_batches,
          exp_ckpt_pth=args.exp_ckpt_pth, eff_start_epoch=eff_start_epoch, eff_time_start=eff_time_,)

def train(model, train_loader, val_loader, args, gpu, 
          train_sampler, param_groups, wandb_logger, stats_file, lambd, inv_constr_keys,
          proj_only=False, epochs=20, warm_up_epochs=5,
          log_freq_step=1, eval_freq_step=1, n_val_batches=100, exp_ckpt_pth=None,
          eff_start_epoch=0, eff_time_start=0):
    
    # DDP
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # Optimizer
    optimizer = build_optimizer(args, model, param_groups)

    # Load checkpoint, if available
    def load_ckpt(ckpt_pth):
        if args.rank == 0:
            print(f"Resuming from checkpoint {ckpt_pth}")
        ckpt = torch.load(ckpt_pth, map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt["epoch"]

    exp_ckpt_pth = args.exp_ckpt_pth if exp_ckpt_pth is None else exp_ckpt_pth    
    if os.path.isfile(exp_ckpt_pth):
        # Resume from checkpoint path with same name as current experiment
        start_epoch = load_ckpt(exp_ckpt_pth) + 1
    else:
        start_epoch = 0
    
    # Learning rate scheduling
    eff_start_step = eff_start_epoch * len(train_loader)
    max_steps = epochs * len(train_loader)
    warmup_steps = warm_up_epochs * len(train_loader)
    base_lr_bb = args.base_lr_bb * args.batch_size / 256.
    if proj_only:
        base_lr_proj = args.base_lr_proj_only * args.batch_size / 256.
    else:
        base_lr_proj = args.base_lr_proj * args.batch_size / 256.

    warmup_steps_lambda = warm_up_epochs * len(train_loader)
    lr_lambda = base_lr_lambda = args.base_lr_lambda * args.batch_size / 256.

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    print("Setup complete. Starting to train...")

    # Test correct weights are being updated
    if args.debug:
        bb_param, proj_param_first, proj_param_last = None, None, None
        for p in model.module.backbone.parameters():
            bb_param = p[0][0][0][0].item()
            break
        for p in list(model.module.projector)[0].parameters():
            proj_param_first = p[-1][0].item()
            break
        for p in list(model.module.projector)[-1].parameters():
            proj_param_last = p[-1][0].item()
            break
        if model.module.projector_style is not None:
            for p in list(model.module.projector_style)[0].parameters():
                proj_param_style = p[-1][0].item()
                break
            
    for epoch in range(start_epoch, epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for step, input_batch in enumerate(train_loader, start=epoch * len(train_loader)):
            # Forward pass
            if args.not_fp16:
                loss, logs = model.forward(make_inputs(input_batch, gpu), lambd=lambd)
            else:
                with torch.cuda.amp.autocast():
                    loss, logs = model.forward(make_inputs(input_batch, gpu), lambd=lambd)
            
            # Outer/dual step (\lambda) after warmup_steps_lambda if adaptive (base_lr_lambda > 0)
            if base_lr_lambda > 0 and step >= warmup_steps_lambda:
                with torch.no_grad():
                    lr_lambda = adjust_learning_rate(step, max_steps, warmup_steps_lambda, base_lr_lambda, args.end_lr_ratio)
                    constraint_values = [model.module.inv_constraint_loss(logs[k].item(), space=i) 
                                         for i, k in enumerate(inv_constr_keys)]
                    lambd = update_lambdas(lambd, constraint_values, lr_lambda, args.tolerance, args.adapt_cont_lambd)

            # Inner/primal step (\theta)
            if proj_only:
                lr_proj = adjust_learning_rate(step, max_steps, warmup_steps, base_lr_proj, args.proj_only_end_lr_ratio)
                lr_bb = 0
            else:
                lr_proj = adjust_learning_rate(step, max_steps, warmup_steps, base_lr_proj, args.end_lr_ratio)
                lr_bb = adjust_learning_rate(step, max_steps, warmup_steps, base_lr_bb, args.end_lr_ratio)
            
            for param_group in optimizer.param_groups:
                name = param_group["name"]
                if name == "backbone":
                    param_group["lr"] = lr_bb
                elif "projector" in name:        # projector and projector_style
                    param_group["lr"] = lr_proj
                elif name == "classifier_h":
                    param_group["lr"] = lr_bb
                elif "classifier_z" in name:     # classifier_z, classifier_z_style and classifier_z_joint
                    param_group["lr"] = lr_proj
                else:
                    raise ValueError(f"Unknown param group name: {name}")

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
                eff_epoch = epoch + eff_start_epoch
                eff_step = step + eff_start_step
                eff_time = int(time.time() - start_time) + eff_time_start
                all_logs = dict(epoch=eff_epoch, step=eff_step, time=eff_time, lr_bb=lr_bb, 
                                lr_proj=lr_proj, lr_lambda=lr_lambda)
                all_logs.update(logs_)      # add afterwards to allow "epoch" first but logs_ to be sorted
                all_logs = {k: round(v, 6) for k, v in all_logs.items()}

                if args.wandb:
                    wandb_logger.log_metrics(all_logs, step=eff_step)
                else:
                    print(json.dumps(all_logs), file=stats_file)

                print(json.dumps(all_logs))

            if args.debug and step >= 2:    # 3 steps for debugging (0,1,2)
                # Test correct weights are being updated
                for p in model.module.backbone.parameters():
                    print("Backbone weights updated:",  abs(bb_param - p[0][0][0][0].item()) > 0)
                    break
                for p in list(model.module.projector)[0].parameters():
                    print("Projector first-layer weights updated:", abs(proj_param_first - p[-1][0].item()) > 0)
                    break
                for p in list(model.module.projector)[-1].parameters():
                    print("Projector final-layer weights updated:", abs(proj_param_last - p[-1][0].item()) > 0)
                    break
                if model.module.projector_style is not None:
                    for p in list(model.module.projector_style)[0].parameters():
                        print("Projector-style updated:", abs(proj_param_style - p[-1][0].item()) > 0)
                        break
                break
        
        # Save checkpoints at regular intervals, overwriting previous ones
        if args.rank == 0 and (((epoch + 1) == epochs) or ((epoch + 1) % args.ckpt_freq == 0)):
            save_ckpt(model, optimizer, args, epoch, exp_ckpt_pth)
        
        if args.debug:  # single epoch for debugging
            break
    print("Training complete.")

    return model.module, lambd

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
        mlp_style_structure=args.mlp_style,
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


def update_lambdas(lambdas, constraint_values, lr, tolerance, adapt_cont_lambd=True):
    lambdas_out = torch.empty_like(lambdas)         # avoid inplace ops
    for i in range(len(lambdas)):                   # don't update/adapt lambda_0
        if i == 0 and not adapt_cont_lambd:
            lambdas_out[i] = lambdas[i]
            continue
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
