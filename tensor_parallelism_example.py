import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.labels = torch.randint(0, 2, (length, 1)).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.len

class SimpleLLM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLLM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, epochs=5):
    setup(rank, world_size)
    dataset = DummyDataset(10, 1000)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                           num_replicas=world_size,
                                                           rank=rank,
                                                           shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = SimpleLLM(10, 256, 1).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    for epoch in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(rank), labels.to(rank)
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % 10 == 0 and rank == 0:
                print(f'Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

    cleanup()

# Note: Requires to be launched with torch.distributed.launch or torchrun for multiple GPUs
# Example:
# torchrun --nproc_per_node=2 tensor_parallelism_example.py

