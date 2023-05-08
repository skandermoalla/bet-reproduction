import hydra
import json
import numpy as np
import utils.metrics
import wandb
import omegaconf
from pathlib import Path
from omegaconf import OmegaConf


def get_train_and_eval_log(snapshot_dir):
    """
    Get the train and eval config from the wandb config file.
    Args:
        snapshot_dir (pathlib.Path): Path to the snapshot directory.
    Returns:
        train_config (dict): Dictionary with the train config.
    """
    local_path_to_wandb_config = Path("wandb", "latest-run", "files", "config.yaml")
    wandb_config_path = snapshot_dir / local_path_to_wandb_config

    wandb_config = OmegaConf.load(str(wandb_config_path))
    train_config = OmegaConf.to_container(
        wandb_config["train_config"]["value"], resolve=True
    )

    eval_config = {}
    for key, value in wandb_config.items():
        if key != "train_config":
            if type(value) != omegaconf.dictconfig.DictConfig:
                eval_config[key] = value
            else:
                eval_config[key] = OmegaConf.to_container(value, resolve=True)
    return train_config, eval_config


def get_environment_name(train_config):
    """Get the name of the environment from the train config."""
    return train_config["env"]["name"]


def process_tags(tags):
    """Process the tags from the wandb config.
    Args:
        tags (str): String with the tags separated by commas.
    Returns:
        tags (tuple): Tuple with the tags."""
    return tuple(tags.split(","))


def check_eval_vectorized(eval_config):
    """Check if the evaluation was vectorized."""
    return eval_config["experiment"]["value"]["vectorized_env"]


def get_snapshot_metrics(obs_path, env_name, eval_was_vectorized):
    """
        Compute the metrics for the snapshot.
    Args:
        file_name (str): Path to the snapshot file.
        env_name (str): Name of the environment.
        eval_was_vectorized (bool): If the evaluation was vectorized.
    Returns:
        snapshot_metrics (tuple): Tuple with the metrics for the snapshot.
    """
    # Load the rollout
    rollout = np.load(str(obs_path), allow_pickle=True)

    # Compute metrics for blockpush
    if env_name == "blockpush":

        if eval_was_vectorized:
            # Load done at
            done_at_path = obs_path.parent / "done_at.npy"
            done_at_array = np.load(str(done_at_path), allow_pickle=True)
        else:
            done_at_array = None

        (
            prob_metrics,
            abs_metrics,
            reward_list,
        ) = utils.metrics.compute_blockpush_metrics(
            rollout,
            eval_was_vectorized=eval_was_vectorized,
            done_at_array=done_at_array,
            tolerance=0.05,
        )
        env_specific_metrics = {**prob_metrics, **abs_metrics}

    # Compute metrics for kitchen
    elif env_name == "kitchen":
        (
            elements,
            mappings,
            timesteps,
        ) = utils.metrics.compute_kitchen_sequences(rollout, treshhold=0.3)
        # Compute the sequence entropy
        count_seq, sequence_entropy = utils.metrics.compute_sequence_entropy(mappings)
        # Compute the task entropy
        elements_dict = utils.metrics.compute_sequence_dict(elements)
        task_entropy = utils.metrics.compute_task_entropy(elements_dict)
        # Compute table 1 metrics
        table1_metrics, reward_list = utils.metrics.compute_kitchen_metrics(elements)
        env_specific_metrics = {
            "sequence_entropy": sequence_entropy,
            "task_entropy": task_entropy,
            **table1_metrics,
        }

    return {
        **env_specific_metrics,
        "average_reward": sum(reward_list) / len(reward_list),
    }


@hydra.main(version_base="1.2", config_path="configs", config_name="config_metrics")
def main(cfg):
    """
    Compute the metrics for the snapshots.
    Args:
        cfg (omegaconf.dictconfig.DictConfig): Configuration for the metrics.
    """
    work_dir = Path.cwd()
    print(f"Saving to {work_dir}.")

    load_dir = Path(cfg.load_dir)

    # Ensure that the dir provided is correct
    if len(list(load_dir.glob("*"))) == 0:
        raise Exception("The provided directory is empty")

    for subdir in load_dir.glob("*"):
        # Get in snapshot directories
        if "snapshot_" in str(subdir):
            print(f" === Starting {subdir.stem} === ")

            snapshot_dir = load_dir / subdir
            train_config, eval_config = get_train_and_eval_log(snapshot_dir)
            tags = process_tags(cfg.tags)

            wandb_run = wandb.init(
                dir=work_dir,
                project=cfg.project,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
            )
            wandb.config.update(
                {
                    "train_config": train_config,
                    "eval_config": eval_config,
                }
            )
            wandb_run.tags += tags

            env_name = get_environment_name(train_config)
            eval_was_vectorized = check_eval_vectorized(eval_config)
            obs_path = snapshot_dir / "obs_trajs.npy"

            snapshot_metrics = get_snapshot_metrics(
                obs_path=obs_path,
                env_name=env_name,
                eval_was_vectorized=eval_was_vectorized,
            )

            # Store computed metrics
            snapshot_name = snapshot_dir.stem
            with open(snapshot_name + "_metrics", "w") as f:
                json.dump(snapshot_metrics, f, indent=4)

            wandb.log(snapshot_metrics)
            wandb.finish()

            print(f" === Finished {subdir.stem} ===\n")


if __name__ == "__main__":
    main()
