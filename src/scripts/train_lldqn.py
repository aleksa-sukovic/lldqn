import os
import sys
import wandb

from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter


# 1. Ensures modules are loaded. This assumes script is run
#    from the root of the repository. Example:
#    python src/scripts/train_lldqn.py
if os.path.abspath(os.path.join('./src')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join('./src')))

from models import Task, ControlNetwork, KnowledgeBase
from models.trainer import DQNTrainer
from models.wrappers import EncodeObservation


# 2. Define configuration variables. In addition, define
#    static data, such as the list of tasks.
WANDB_PROJECT = "lldqn"
WANDB_LOG_DIR = "./src/data"
WANDB_TENSORBOARD = "./src/data/tensorboard"

# 3. Experiment I: Training "Acrobot-v1" with knowledge gained from "CartPole-v1".
cart_pole = Task(
    env_name="CartPole-v1",
    env_model=ControlNetwork,
    save_data_dir="./src/data/models",
    use_baseline=True,
    wrappers=[(EncodeObservation, dict())],
    version=3,
)
cart_pole.load(policy=True)
knowledge_base = KnowledgeBase(tasks=[cart_pole])
task = Task(
    env_name="Acrobot-v1",
    env_model=ControlNetwork,
    save_data_dir="./src/data/models",
    save_model_name="Acrobot-v1-Policy-LLDQN",
    use_baseline=False,
    knowledge_base=knowledge_base,
    wrappers=[(EncodeObservation, dict())],
)

for repeat in range(3):
    task = Task(
        env_name="Acrobot-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
        save_model_name=f"Acrobot-v1-Policy-LLDQN-{repeat + 1}.pt",
        use_baseline=False,
        knowledge_base=knowledge_base,
        wrappers=[(EncodeObservation, dict())],
        version = repeat + 1,
    )

    wandb.init(
        project=WANDB_PROJECT,
        dir=WANDB_LOG_DIR,
        group=task.name,
        job_type="LLDQN-Policy-Train",
        name=task.save_model_name,
        sync_tensorboard=True,
        reinit=True,
        monitor_gym=True,
        config={
            "train/repeat": repeat,
        }
    )
    wandb.define_metric("train/reward", summary="mean")
    wandb.define_metric("test/reward", summary="mean")

    logger = WandbLogger()
    logger.load(SummaryWriter(WANDB_TENSORBOARD))

    task.load()
    trainer = DQNTrainer(task, logger=logger)
    result = trainer.run()

    print("Finished repeat {}. Time taken: {:.4}s".format(repeat + 1, result["duration"]))


# 4. Experiment II: Training "CartPole-v1" with knowledge gained from "Acrobot-v1".
acrobot = Task(
    env_name="Acrobot-v1",
    env_model=ControlNetwork,
    save_data_dir="./src/data/models",
    use_baseline=True,
    wrappers=[(EncodeObservation, dict())],
    version=3,
)
acrobot.load(policy=True)
knowledge_base = KnowledgeBase(tasks=[acrobot])

for repeat in range(3):
    task = Task(
        env_name="CartPole-v1",
        env_model=ControlNetwork,
        save_data_dir="./src/data/models",
        save_model_name=f"CartPole-v1-Policy-LLDQN-{repeat + 1}.pt",
        use_baseline=False,
        knowledge_base=knowledge_base,
        wrappers=[(EncodeObservation, dict())],
        version = repeat + 1,
    )

    wandb.init(
        project=WANDB_PROJECT,
        dir=WANDB_LOG_DIR,
        group=task.name,
        job_type="LLDQN-Policy-Train",
        name=task.save_model_name,
        sync_tensorboard=True,
        reinit=True,
        monitor_gym=True,
        config={
            "train/repeat": repeat,
        }
    )
    wandb.define_metric("train/reward", summary="mean")
    wandb.define_metric("test/reward", summary="mean")

    logger = WandbLogger()
    logger.load(SummaryWriter(WANDB_TENSORBOARD))

    task.load()
    trainer = DQNTrainer(task, logger=logger)
    result = trainer.run()

    print("Finished repeat {}. Time taken: {:.4}s".format(repeat + 1, result["duration"]))
