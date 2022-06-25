import os
import sys
import wandb
import torch
import numpy as np

from torch.utils.data import DataLoader

# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/train_autoencoder.py
if os.path.abspath(os.path.join('./src')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('./src')))

from models import Task, ControlNetwork, Autoencoder
from utils import get_observation_dataset, get_action_dataset

# 2. Define configuration variables. In addition, define
#    static data, such as the list of tasks.
CODE_DIM = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH = 128
TRAIN_OBSERVATION_EPOCHS = 250
TRAIN_ACTION_EPOCHS = 50
TRAIN_LEARNING_RATE = 1e-3
TRAIN_WEIGHT_DECAY = 1e-6
TRAIN_EXPLORATION_EPISODES = 200
WANDB_PROJECT = "lldqn"
WANDB_LOG_DIR = "./src/data"
WANDB_GROUP_OBSERVATION="Observation Autoencoder"
WANDB_GROUP_ACTION="Action Autoencoder"
WANDB_JOB_TYPE="training"
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

# 3. Define helper functions.
def train_observation_representation(task: Task):
    dataset = get_observation_dataset(task)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH, shuffle=True)
    encoder = Autoencoder(
        input_dim=np.prod(task.state_shape),
        code_dim=CODE_DIM,
        save_data_dir=task.save_data_dir,
        save_model_name=f"{task.name}-Observation-Autoencoder.pt",
    )
    encoder.train_encoder(dataloader, TRAIN_OBSERVATION_EPOCHS, TRAIN_LEARNING_RATE, TRAIN_WEIGHT_DECAY)

def train_action_representation(task: Task):
    dataset = get_action_dataset(task, n_sample=10000)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH, shuffle=True)
    encoder = Autoencoder(
        input_dim=np.prod(task.action_shape),
        code_dim=CODE_DIM,
        save_data_dir=task.save_data_dir,
        save_model_name=f"{task.name}-Action-Autoencoder.pt",
    )
    encoder.train_encoder(dataloader, TRAIN_ACTION_EPOCHS, TRAIN_LEARNING_RATE, TRAIN_WEIGHT_DECAY)

# 4. Train invariant observation representation for each defined task.
for task in TASKS:
    wandb.init(
        project=WANDB_PROJECT,
        dir=WANDB_LOG_DIR,
        group=WANDB_GROUP_OBSERVATION,
        job_type=WANDB_JOB_TYPE,
        reinit=True,
        name=task.name,
    )

    train_observation_representation(task)

# 5. Train invariant action representation for each defined task.
for task in TASKS:
    wandb.init(
        project=WANDB_PROJECT,
        dir=WANDB_LOG_DIR,
        group=WANDB_GROUP_ACTION,
        job_type=WANDB_JOB_TYPE,
        reinit=True,
        name=task.name,
    )

    train_action_representation(task)
