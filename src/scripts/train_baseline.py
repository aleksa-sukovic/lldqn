import os
import sys
import wandb

from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter


# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/train_invariant_representation.py
if os.path.abspath(os.path.join('./src')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('./src')))

from models import Task, ControlNetwork
from models.trainer import DQNTrainer
from models.wrappers import EncodeObservation


# 2. Define configuration variables. In addition, define
#    static data, such as the list of tasks.
WANDB_PROJECT = "lldqn"
WANDB_LOG_DIR = "./src/data"
WANDB_TENSORBOARD = "./src/data/tensorboard"
TASKS = [
    dict(
        env_name="Acrobot-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
        use_baseline=True,
        wrappers=[(EncodeObservation, dict())]
    ),
    dict(
        env_name="CartPole-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
        use_baseline=True,
        wrappers=[(EncodeObservation, dict())]
    ),
]

# 4. Train each task in a sequence.
for task_data in TASKS:
    for repeat in range(3):
        task = Task(**task_data, version = repeat + 1)

        wandb.init(
            project=WANDB_PROJECT,
            dir=WANDB_LOG_DIR,
            group=task.name,
            job_type="Policy-Train",
            name=task.save_model_name,
            sync_tensorboard=True,
            reinit=True,
            monitor_gym=True,
            config={
                "train/repeat": repeat,
            }
        )

        logger = WandbLogger()
        logger.load(SummaryWriter(WANDB_TENSORBOARD))

        task.load()
        trainer = DQNTrainer(task, logger=logger)
        result = trainer.run()

        print("Finished repeat {}. Time taken: {:.4}s".format(repeat + 1, result["duration"]))
