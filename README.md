# Overview
This is the code repository our paper ["Self-Supervised Disentanglement by Leveraging Structure in Data Augmentations"](https://arxiv.org/abs/2311.08815).

<p align="center">
  <img src="https://github.com/cianeastwood/ssl_disentangled/blob/main/assets/ssl_disent_overview.png?raw=true" width="500" alt="SSL Overview" />
</p>

## What's in this repo?
- The main code snippets and modules for our self-supervised method.
- Some brief instructions on how to run experiments.

## What's not in this repo?
- Clean, easy-to-use code with clear instructions (yet!).
- The code for evaluating the trained models on downstream tasks: we adapted [this code](https://github.com/linusericsson/ssl-transfer)!


## Useful files
- `algorithms.py`: Contains our ssl algorithms as well as the baselines, making it clear how we a. route samples to each space and b. compute the loss.
- `augmentations.py`: Contains our structured data-augmentation procedure.
- `main.py`: The main script for running experiments.
- `main_pretrained.py`: The main script for running experiments with a pretrained model (i.e., using our method to fine-tune a base model).


## Running a test command
To ensure the environment has been set up correctly (instructions coming soon...) you can run the following test command from the base directory:

```sh
python -m torch.distributed.launch --nproc_per_node=2 main.py --exp-dir=/checkpoint/ceastwood/ssl_disent/test \
--alg=vicreg --aug-type=asymmetric --dataset=imagenet --debug --batch-size=32 --base-lr-lambda=0.1 --tolerance=0.0001 \
--warm-up-epochs-lambda=0 --view-samples --sweep-name=test
```

## Questions?
- Please reach out!