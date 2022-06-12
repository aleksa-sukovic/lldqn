import os
import sys
import wandb

# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/train_invariant_representation.py
if os.path.abspath(os.path.join('./src')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('./src')))

from models import Task, AugmentObservationSpaceWrapper


# 2. Define configuration variables. In addition, define
#    static data, such as the list of tasks.
WANDB_PROJECT = "lldqn"
WANDB_LOG_DIR = "./src/data"
WANDB_GROUP="invariant_representation"
WANDB_JOB_TYPE="training"
TASKS = [
    Task(
        env_name="Acrobot-v1",
        wrappers=[AugmentObservationSpaceWrapper],
        save_data_dir="./src/data/models",
    ),
    Task(
        env_name="MountainCarContinuous-v0",
        wrappers=[AugmentObservationSpaceWrapper],
        save_data_dir="./src/data/models",
    ),
    Task(
        env_name="MountainCar-v0",
        wrappers=[AugmentObservationSpaceWrapper],
        save_data_dir="./src/data/models",
    ),
    Task(
        env_name="Pendulum-v1",
        wrappers=[AugmentObservationSpaceWrapper],
        save_data_dir="./src/data/models",
    ),
    Task(
        env_name="CartPole-v1",
        wrappers=[AugmentObservationSpaceWrapper],
        save_data_dir="./src/data/models",
    )
]

# 3. Train invariant representations for each defined task.
for task in TASKS:
    wandb.init(
        project=WANDB_PROJECT,
        dir=WANDB_LOG_DIR,
        group=WANDB_GROUP,
        job_type=WANDB_JOB_TYPE,
        reinit=True,
        name=task.name,
    )
    task.compile()
