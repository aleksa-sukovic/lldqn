import os
import sys
import wandb


# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/evaluate_lldqn.py
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
    dict(
        env_name="Acrobot-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
        use_baseline=False,
        wrappers=[(EncodeObservation, dict())],
    ),
    dict(
        env_name="CartPole-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
        use_baseline=False,
        wrappers=[(EncodeObservation, dict())],
    ),
]

# 4. Train each task in a sequence.
for task_data in TASKS:
    for version in range(3):
        task = Task(**task_data, version=version + 1, save_model_name=f'{task_data["env_name"]}-Policy-LLDQN-{version + 1}.pt')

        wandb.init(
            project=WANDB_PROJECT,
            dir=WANDB_LOG_DIR,
            group=task.name,
            job_type=f"Evaluate-LLDQN",
            name=f"{task.name}-Version-{task.version}",
            sync_tensorboard=True,
            monitor_gym=True,
            reinit=True,
            config={
                "test/episodes": 50,
                "test/noise": False,
            }
        )

        task.load()
        result = task.collector_test.collect(n_episode=50)

        wandb.run.summary["test/episodes"] = result["n/ep"]
        wandb.run.summary["test/reward"] = result["rew"]
        wandb.run.summary["test/reward_std"] = result["rew_std"]
