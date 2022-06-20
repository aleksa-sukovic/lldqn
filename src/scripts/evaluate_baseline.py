import os
import sys
import wandb
import numpy as np

from gym.wrappers.pixel_observation import PixelObservationWrapper


# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/train_invariant_representation.py
if os.path.abspath(os.path.join('./src')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('./src')))

from models import Task, ControlNetwork
from models.wrappers import EncodeObservation


# 2. Define configuration variables. In addition, define
#    static data, such as the list of tasks.
WANDB_PROJECT = "lldqn"
WANDB_LOG_DIR = "./src/data"
WANDB_TENSORBOARD = "./src/data/tensorboard"
WANDB_GROUP="baseline"
WANDB_JOB_TYPE="evaluation"
TASKS = [
    Task(
        env_name="Acrobot-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
        use_baseline=True,
        wrappers=[(EncodeObservation, dict())]
    ),
]

# 4. Train each task in a sequence.
for task in TASKS:
    wandb.init(
        project=WANDB_PROJECT,
        dir=WANDB_LOG_DIR,
        group=task.name,
        job_type="Policy-Evaluate",
        name=task.save_model_name,
        sync_tensorboard=True,
        monitor_gym=True,
        reinit=True,
        config={
            "test/episodes": 20,
            "test/noise": False,
        }
    )

    task.load()
    result = task.collector_test.collect(n_episode=5, render = 1 / 35)

    wandb.run.summary["test/episodes"] = result["n/ep"]
    wandb.run.summary["test/reward"] = result["rew"]
    wandb.run.summary["test/reward_std"] = result["rew_std"]
