import torch

from gym import Env, ObservationWrapper

from models import Task


class EncodeObservation(ObservationWrapper):
    def __init__(self, env: Env, task: Task) -> None:
        super().__init__(env)
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def observation(self, observation):
        result = torch.tensor(observation, device=self.device)
        result = self.task.observation_encoder.encoder(result)
        result = result.cpu().detach().numpy()
        return result
