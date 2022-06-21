import os
import sys
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from typing import List

# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/evaluate_autoencoder.py
if os.path.abspath(os.path.join('./src')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('./src')))

from models import Task, ControlNetwork


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
    ),
    Task(
        env_name="CartPole-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
    ),
    Task(
        env_name="MountainCarContinuous-v0",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
    ),
    Task(
        env_name="MountainCar-v0",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
    ),
    Task(
        env_name="Pendulum-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
    ),
]

# 4. Define helper functions.
def make_similarity_matrix(tasks: List[Task]):
    data = np.zeros((len(tasks), len(tasks)))

    for i, task_a in enumerate(TASKS):
        for j, task_b in enumerate(TASKS):
            data[i, j] = task_a.get_similarity_index(task_b)

    data_frame = pd.DataFrame(data, index = [t.name for t in TASKS], columns = [t.name for t in TASKS])
    plt.figure(figsize = (10, 7))
    sn.heatmap(data_frame, annot=True, cmap="OrRd_r")
    plt.show()


# 5. Generate and plot results.
make_similarity_matrix(TASKS)
