'''
Model training pipeline scaffold, not used later
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score


# Here we replace the SimpleNN model with an import of our model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Currently a temporary function to generate date, will be replaced by our images.
def generate_data():
    data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100, 1), dtype=torch.float32)
    return data, labels

def train(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Loss/train', loss.item(), global_step=global_step)


        predictions = (output > 0.5).float()  # Assuming a threshold of 0.5 for binary classification
        accuracy = accuracy_score(target.cpu().numpy(), predictions.cpu().numpy())
        writer.add_scalar('Accuracy/train', accuracy, global_step=global_step)

        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = accuracy_score(all_targets, all_predictions)
    print(f'Train Epoch: {epoch}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}')


def main():
    writer = SummaryWriter(log_dir='logs')

    train_data, train_labels = generate_data()
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = SimpleNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, epoch, writer)

    torch.save(model.state_dict(), 'simple_model.pth') # This is model is saved into PyTorch file

    writer.close()

if __name__ == "__main__":
    main()
