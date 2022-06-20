import tqdm
import torch
import wandb

from os.path import join, exists
from torch import nn
from torch.utils.data import DataLoader


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        code_dim: int = 3,
        save_data_dir: str = "",
        save_model_name: str = "",
        load: bool = False,
    ):
        super().__init__()
        self.save_data_dir = save_data_dir
        self.save_model_name = save_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init(input_dim, code_dim)
        self.to(self.device)
        if load:
            self.load()

    def init(self, input_dim: int, code_dim: int):
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, code_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Tanh(),
        )

    def load(self):
        if not exists(join(self.save_data_dir, self.save_model_name)):
            return
        payload = torch.load(join(self.save_data_dir, self.save_model_name))
        self.init(payload["input_dim"], payload["code_dim"])
        self.load_state_dict(payload["model"])
        self.eval()

    def save(self, model: nn.Module):
        payload = {
            "model": model.state_dict(),
            "input_dim": self.input_dim,
            "code_dim": self.code_dim,
        }
        torch.save(payload, join(self.save_data_dir, self.save_model_name))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_encoder(
        self,
        dataloader: DataLoader,
        train_epochs: int = 250,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ) -> None:
        model = self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        progress = tqdm.tqdm(total=train_epochs)
        wandb.config.update({
            "epochs": train_epochs,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        })

        for _ in range(train_epochs):
            for data in dataloader:
                batch = data[0]
                batch = batch.to(self.device)

                output = model(batch)
                loss = criterion(output, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            progress.update()
            progress.set_postfix(loss="{:.4f}".format(loss.item()))
            wandb.log({"train/loss": loss.item()})

        self.save(model)
