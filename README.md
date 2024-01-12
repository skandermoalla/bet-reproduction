# A Reproducibility study of [_Behavior Transformers: Cloning k modes with one stone_](https://github.com/notmahi/bet)

## Overview

This repository extends the original code repository of [_Behavior Transformers: Cloning k modes with one
stone_](https://github.com/notmahi/bet) to serve as an in-depth reproducibility assessment of the paper.

It contains scripts to reproduce each of the figures and tables in the paper in addition to scripts to run additional
experiments.

It is structured as follows:

## Table of contents

<!-- TOC -->
* [A Reproducibility study of _Behavior Transformers: Cloning k modes with one stone_](#a-reproducibility-study-of-behavior-transformers-cloning-k-modes-with-one-stone)
  * [Overview](#overview)
  * [Table of contents](#table-of-contents)
  * [Getting started](#getting-started)
    * [Installation and setup](#installation-and-setup)
      * [Cloning](#cloning)
      * [Datasets](#datasets)
    * [Environment](#environment)
      * [A&B: amd64 (CUDA and CPU-only)](#ab--amd64--cuda-and-cpu-only-)
        * [Using a pre-built Docker Image](#using-a-pre-built-docker-image)
      * [C: MPS (Apple silicon)](#c--mps--apple-silicon-)
    * [Logging](#logging)
    * [Testing](#testing)
    * [Using the reproducibility scripts](#using-the-reproducibility-scripts)
  * [Reproducing our results](#reproducing-our-results)
  * [Experiment with different configurations](#experiment-with-different-configurations)
<!-- TOC -->


## Getting started

### Installation and setup

#### Cloning

Clone the repository with its submodules.
We track [Relay Policy Learning](https://github.com/google-research/relay-policy-learning) repo as a submodule for the
Franka kitchen environment.
It uses [`git-lfs`](https://git-lfs.github.com/). Make sure you have it installed.

```bash
git clone --recurse-submodules
```

If you didn't clone the repo with `--recurse-submodules`, you can clone the submodules with:

```bash
git submodule update --init
```

#### Datasets

The datasets are stored in the `data` folder and are not tracked by `git`.

1. Download the datasets [here](https://osf.io/download/4g53p/).
   ```bash
   wget https://osf.io/download/4g53p/ -O ./data/bet_data_release.tar.gz
   ```
2. Extract the datasets into the `data/` folder.

   ```bash
   tar -xvf data/bet_data_release.tar.gz -C data
   ```

The contents of the `data` folder should look like this:

* `data/bet_data_release.tar.gz`: The archive just downloaded.
* `data/bet_data_release`: contains the datasets released by the paper authors.
* `data/README.md`: A placeholder.

### Environment

We provide installation methods to meet different systems.
The methods aim to insure ease of use, portability, and reproducibility thanks to Docker.
It is hard to cover all systems, so we focused on the main ones.

- A: **amd64 with CUDA:** for machines with Nvidia GPUs with Intel CPUs.
- B: **amd 64 CPU-only:** for machines with Intel CPUs.
- C: **arm64 with MPS:** to leverage the M1 GPU of Apple machines.

#### A&B: amd64 (CUDA and CPU-only)

This installation method is adapted from the [Cresset initiative](https://github.com/cresset-template/cresset).
Refer to the Cresset repository for more details.

Steps prefixed with [CUDA] are only required for the CUDA option.

**Prerequisites:**
To check if you have each of them run `<command-name> --version` or `<command-name> version` in the terminal.

* [`make`](https://cmake.org/install/).
* [`docker`](https://docs.docker.com/engine/). (v20.10+)
* [`docker compose`](https://docs.docker.com/compose/install/) (V2)
* [CUDA] [Nvidia CUDA Driver](https://www.nvidia.com/download/index.aspx) (Only the driver. No CUDA toolkit, etc)
* [CUDA] [`nvidia-docker`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) (the NVIDIA Container Toolkit).

**Installation**

```bash
cd installation/amd64
```

Run

```bash
make env
```

The `.env` file will be created in the installation directory. You need to edit it according to the following needs:

[CUDA] If you are using an old Nvidia GPU (i.e. [capability](https://developer.nvidia.com/cuda-gpus#compute)) < 3.7) you
need to compile PyTorch from source.
Find the compute capability for your GPU and edit it below.

```bash
BUILD_MODE=include               # Whether to build PyTorch from source.
CCA=3.5                          # Compute capability.
```

[CUDA] If your Nvidia drivers are also old you may need to change the CUDA Toolkit version.
See
the [compatibility matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)
for compatible versions of the CUDA driver and CUDA Toolkit

```bash
CUDA_VERSION=11.3.1                # Must be compatible with hardware and CUDA driver.
CUDNN_VERSION=8                    # Only major version specifications are available.
```

Build the docker image by running the following.
Set `SERVICE=cuda` or `SERVICE=cpu` for your desired option. The default is `SERVICE=cuda`.

```bash
make build SERVICE=<cpu|cuda>
````

Then to use the environment, run

```bash
make exec SERVICE=<cpu|cuda>
```

and you'll be inside the container.

To run multiple instances of the container you can use

```bash
make run SERVICE=<cpu|cuda>
```

##### Using a pre-built Docker Image

We provide pre-built AMD64 CUDA images on Docker Hub for easier reproducibility.
Visit [https://hub.docker.com/r/mlrc2022bet/bet-reproduction/tags](https://hub.docker.com/r/mlrc2022bet/bet-reproduction/tags)
to find a suitable Docker image.

To download and use the CUDA 11.2.2 image, follow these steps.
1. Run `docker pull mlrc2022bet/bet-reproduction:cuda112-py38` on the host.
2. Run `docker tag mlrc2022bet/bet-reproduction:cuda112-py38 bet-user:cuda` to change the name of the image.
This part is necessary due to the naming conventions in the `docker-compose.yaml` file.
3. Edit the `.env` file to set the `UID, GID, USR, GRP, and IMAGE_NAME` values generated by the `make env` command.

An issue with the Docker Hub images is that their User ID (UID) and Group ID (GID)
values have been fixed to the default values of `UID=1000, GID=1000, USR=user, GRP=user`.

The `.env` file must be edited to use the downloaded images.
An example of the resulting `.env` file would be as follows:
```text
# Make sure that `IMAGE_NAME` matches `bet-user:cuda` instead of the default value generated by `make env`.
IMAGE_NAME=bet-user  # Do not add `:cuda` to the name.
PROJECT_ROOT=/opt/project
UID=1000
GID=1000
USR=user
GRP=user
# W&B configurations must be included manually.
WANDB_MODE=online
WANDB_API_KEY=<your_key>
```

4. Run `make up` to start the container.
5. Run `make exec` to enter the container.
6. Run `sudo chown -R $(id -u):$(id -g) /opt/project` inside the container to change ownership of the project root.
7. After finishing training, run `sudo chown -R $(id -u):$(id -g) bet-reproduction` from the host to restore ownership to the host user.


#### C: MPS (Apple silicon)

As the MPS backend isn't supported by PyTorch on Docker, this methods relies on a local installation of `conda`, thus
unfortunately limiting portability and reproducibility.
We provide an `environment.yml` file adapted from the BeT's author's repo to be compatible with the M1 system.

**Prerequisites:**

* `conda`: which we recommend installing with [miniforge](https://github.com/conda-forge/miniforge).

**Installation**

```bash
conda env create --file=installation/osx-arm64/environment.yml
conda activate behavior-transformer
```

Set environment variables.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/relay-policy-learning/adept_envs
export ASSET_PATH=$(pwd)
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
```

### Logging

We track our experiments with [Weights and Biases](https://wandb.ai/site).
To use it, either

1. [Docker] Add your `wandb` [API key](https://wandb.ai/authorize) to the `.env` file
    ```bash
    WANDB_MODE=online
    WANDB_API_KEY=<your_key>
    ```
   then `make up SERVICE=<cpu|cuda>`.

2. Or, run

    ```bash
    export WANDB_MODE=online && wandb login
    ```
   in the docker container, or in your custom environment.

### Testing

Test your setup by running the default training and evaluation scripts in each of the environments.

Environment values (`<env>` below) can be `blockpush`, `kitchen`, `pointmass1`, or `pointmass2`.

Training.

```bash
python train.py env=<env> experiment.num_prior_epochs=1
```

Evaluation.
Find the model you just trained in `train_runs/train_<env>/<date>/<job_id>`.
Plug it in the command below.

```bash
python run_on_env.py env=<env> experiment.num_eval_eps=1 \
model.load_dir=$(pwd)/train_runs/train_<env>/<date>/<job_id>
```

### Using the reproducibility scripts

We provide scripts to reproduce our training and evaluation results. For example,
to reproduce the Blockpush results with the configurations from the paper, run 
`sh reproducibility_scripts/paper_params/blockpush.sh`.
This will run a cross-validation training for 3 independent runs with evaluations for each of the runs.

Other ablations are also available in the `reproducibility_scripts` directory.

Note that the directories in `train_runs` and `eval_runs` directories corresponding to each experiment should not exist before starting to prevent directory name clashes.

For example, the Blockpush experiments using the paper parameters will create 
`train_runs/train_blockpush/reproduction/paper_params` and 
`eval_runs/eval_blockpush/reproduction/paper_params` subdirectories.

The subdirectories inside `reproduction` must be deleted before the corresponding run can be launched.
This may be necessary if the previous run was terminated before completion.

## Reproducing our results

We provide model weights, their rollouts, and their evaluation metrics for all the experiments we ran.
You can use these to reproduce our results at any stage of the pipeline.
In addition, we share our Weights and Biases runs in this [W&B project](https://wandb.ai/skandermoalla/behavior_transformer_repro?workspace=default).

The scripts used to generate the models, their rollouts, and compute their evaluation metrics can be found in `reproducibility_scripts/`

Obtain the model weights, the rollouts, and the logs of the evaluations with

```bash
wget https://www.dropbox.com/s/y7c0cerbjm1hap6/weights_rollouts_and_metrics.tar.gz
tar -xvf weights_rollouts_and_metrics.tar.gz
```

## Experiment with different configurations

The configurations are stored in the `configs/` directory and are subdivided into categories.
They are managed by [Hydra](https://hydra.cc/docs/intro/)
You can experiment with different configurations by passing the relevant flags.
You can get examples on how to do so in the `reproducibility_scripts/` directory.

