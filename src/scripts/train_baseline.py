import os
import sys
import wandb

from gym.wrappers.pixel_observation import PixelObservationWrapper
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter


# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/train_invariant_representation.py
if os.path.abspath(os.path.join('./src')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('./src')))

from models import Task
from models.trainer import DQNTrainer
from models.wrappers import PreprocessObservation, StackObservation

# 2. Define configuration variables. In addition, define
#    static data, such as the list of tasks.
WANDB_PROJECT = "lldqn"
WANDB_LOG_DIR = "./src/data"
WANDB_TENSORBOARD = "./src/data/tensorboard"
TASKS = [
    dict(
        env_name="CartPole-v1",
        save_data_dir="./src/data/models",
        use_baseline=True,
        wrappers=[
            (PixelObservationWrapper, {"pixels_only": False}),
            (PreprocessObservation, {}),
            (StackObservation, {}),
        ],
    ),
    # dict(
    #     env_name="Pendulum-v1",
    #     wrappers=[AugmentObservationSpaceWrapper],
    #     save_data_dir="./src/data/models",
    #     use_baseline=True,
    # ),
    # dict(
    #     env_name="Acrobot-v1",
    #     wrappers=[AugmentObservationSpaceWrapper],
    #     save_data_dir="./src/data/models",
    #     use_baseline=True,
    # ),
    # dict(
    #     env_name="MountainCar-v0",
    #     # wrappers=[AugmentObservationSpaceWrapper],
    #     wrappers=[],
    #     save_data_dir="./src/data/models",
    #     use_baseline=True,
    # ),
    # dict(
    #     env_name="MountainCarContinuous-v0",
    #     wrappers=[AugmentObservationSpaceWrapper],
    #     save_data_dir="./src/data/models",
    #     use_baseline=True,
    # ),
]

# 4. Train each task in a sequence.
for task_data in TASKS:
    for repeat in range(1):
        task = Task(**task_data, version=repeat + 1)

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
                "train/repeat_count": 1,
            }
        )

        logger = WandbLogger()
        logger.load(SummaryWriter(WANDB_TENSORBOARD))
        trainer = DQNTrainer(task, logger=logger)
        result = trainer.run()

        print("Finished repeat {}. Time taken: {:.4}s".format(repeat + 1, result["duration"]))
