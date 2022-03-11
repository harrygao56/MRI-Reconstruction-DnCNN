from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from model import DnCNN


class MRIDataset(Dataset):
    def __init__(self):
        self.xHat = torch.from_numpy(np.load('xHat.npy'))
        self.grndTrth = torch.from_numpy(np.load('grndTrth.npy'))
        self.n_samples = self.xHat.shape[0]

    def __getitem__(self, index):
        return self.xHat[index].float(), self.grndTrth[index].float()

    def __len__(self):
        return self.n_samples


# Create dataset and dataloaders
batch_size = 4
dataset = MRIDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

training_data = torch.utils.data.Subset(dataset, range(0, 220))
training_loader = DataLoader(dataset=training_data, batch_size=batch_size)

valid_data = torch.utils.data.Subset(dataset, range(221, 260))
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)

testing_data = torch.utils.data.Subset(dataset, range(261, 360))
testing_loader = DataLoader(dataset=testing_data, batch_size=batch_size)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN().to(device)

# Loss, optimizer
criterion = nn.MSELoss()
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training info
num_epochs = 500
n_total_steps = len(training_loader)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i, (x, grndTrth) in enumerate(training_loader):
        # Create batch data
        x = x.to(device)
        grndTrth = grndTrth.to(device)

        # Forward Pass
        outputs = model(x)
        loss = criterion(outputs, grndTrth)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Training Loss: {loss.item():.4f}')

    model.eval()
    for i, (x, grndTrth) in enumerate(valid_loader):
        x = x.to(device)
        grndTrth = grndTrth.to(device)

        outputs = model(x.float())
        loss = criterion(outputs, grndTrth)
        print(f'Validation Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "trained_model")
