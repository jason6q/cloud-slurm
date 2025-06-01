"""
    Simple toy example using the Slurm Cluster.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import mlflow
from omegaconf import DictConfig, OmegaConf

# DDP Setup
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

#dist.barrier()

class ToyModel(nn.Module):
    """
        Not really trying anything fancy.
    """
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(3072,768)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(768, 384)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(384, 10)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        y = self.linear1(x)
        y = self.relu1(y)
        y = self.linear2(y)
        y = self.relu2(y)
        y = self.linear3(y)
        out = F.log_softmax(y, dim=1)
        return out

@profile
def main() -> None:
    # Constants, TODO - Move this to a yaml experiment file.
    CIFAR10_PATH = '/mnt/shared-slurm/datasets/CIFAR10'
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda") # Assume cuda.
    EPOCHS = 10

    # SLURM Environment variables
    RANK = int(os.environ["SLURM_PROCID"])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    GPUS_PER_NODE = int(os.environ['SLURM_GPUS_ON_NODE'])
    LOCAL_RANK = RANK - GPUS_PER_NODE * (RANK // GPUS_PER_NODE)
    CPUS_PER_TASK = int(os.environ["SLURM_CPUS_PER_TASK"])
    print(f"RANK = {RANK}\nWORLD_SIZE = {WORLD_SIZE}\nGPUS_PER_NODE = {GPUS_PER_NODE}\nLOCAL_RANK = {LOCAL_RANK}\nCPUS_PER_TASK={CPUS_PER_TASK}")

    ## DDP Init
    setup(RANK, WORLD_SIZE)

    # MLFlow Setup
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
    print(f"Setting MLFlow Tracking URI to {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)

    # Load CIFAR
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
    )
    train_args = {
        "num_workers": CPUS_PER_TASK,
        "pin_memory": True
    }
    test_args = train_args

    train = CIFAR10(root=CIFAR10_PATH, train=True, download=True, transform=transform)
    test = CIFAR10(root=CIFAR10_PATH, train=False, download=False, transform=transform)

    sampler = DistributedSampler(train, num_replicas=WORLD_SIZE, rank=RANK)
    trainloader = DataLoader(train, sampler=sampler, batch_size=BATCH_SIZE, **train_args)
    testloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, **test_args)

    # Init Model
    mod = ToyModel().to(RANK)
    ddp_mod = DDP(mod, device_ids=[RANK])
    optimizer = optim.AdamW(mod.parameters())
    criterion = nn.CrossEntropyLoss()

    # Train Model
    with mlflow.start_run():
        mlflow.log_param('batch_size', BATCH_SIZE)
        mlflow.log_param('epochs', EPOCHS)
        mlflow.log_param('dataset', CIFAR10_PATH)
        running_loss = 0. # Not really trying to do anything fancy; this is just an example.
        total_it = 0
        for e in range(EPOCHS):
            mod.train()
            for i, batch in enumerate(trainloader):
                x,y = batch
                x,y = x.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()

                out = ddp_mod(x)
                loss = criterion(out, y)

                running_loss += loss.item()
                total_it += 1
                if i % 100 == 0:
                    print(f"{e+1} - [{i}/{len(trainloader)}] - {running_loss / total_it}")
                    mlflow.log_metric('training_loss', running_loss / total_it, step=total_it)

                loss.backward()
                optimizer.step()
            
            mod.eval()
            test_loss = 0.
            correct = 0
            with torch.no_grad():
                for i, batch in enumerate(testloader):
                    x,y = batch
                    x,y = x.to(DEVICE), y.to(DEVICE)
                    out = mod(x)
                    loss = criterion(out, y)
                    test_loss += loss.item()
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
            print(f"Test Accuracy: {correct}/{len(test)}")

            mlflow.log_metric('test_loss', test_loss / len(test), step=e)
            mlflow.log_metric('test_pred', correct / len(test), step=e)
    print('Finished Training!')

    ## DDP Destroy
    dist.destroy_process_group()


if __name__ == '__main__':
    main()