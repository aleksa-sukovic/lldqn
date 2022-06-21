import os
import sys
import wandb
import torch
import numpy as np

from torch.utils.data import DataLoader

# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/train_invariant_representation.py
if os.path.abspath(os.path.join('./src')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('./src')))

from models import Task, ControlNetwork, Autoencoder
from utils import get_observation_dataset

# 2. Define configuration variables. In addition, define
#    static data, such as the list of tasks.
CODE_DIM = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH = 128
TRAIN_EPOCHS = 250
TRAIN_LEARNING_RATE = 1e-3
TRAIN_WEIGHT_DECAY = 1e-6
TRAIN_EXPLORATION_EPISODES = 200
WANDB_PROJECT = "lldqn-update"
WANDB_LOG_DIR = "./src/data"
WANDB_GROUP="Observation Autoencoder"
WANDB_JOB_TYPE="training"
TASKS = [
    dict(
        env_name="Acrobot-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
    ),
    dict(
        env_name="CartPole-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
    ),
]

# 3. Define helper functions.
def train_observation_representation(task: Task):
    dataset = get_observation_dataset(task)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH, shuffle=True)
    encoder = Autoencoder(
        input_dim=np.prod(task.state_shape),
        code_dim=CODE_DIM,
        save_data_dir=task.save_data_dir,
        save_model_name=f"{task.name}-Observation-Autoencoder.pt"
    )
    encoder.train_encoder(dataloader, TRAIN_EPOCHS, TRAIN_LEARNING_RATE, TRAIN_WEIGHT_DECAY)

# 4. Train invariant representations for each defined task.
for task_data in TASKS:
    task = Task(**task_data)

    wandb.init(
        project=WANDB_PROJECT,
        dir=WANDB_LOG_DIR,
        group=WANDB_GROUP,
        job_type=WANDB_JOB_TYPE,
        reinit=True,
        name=task.name,
    )

    train_observation_representation(task)
