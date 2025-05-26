import argparse, os, torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--model_out', type=str, default='cnn_lime.pth')
args = parser.parse_args()

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
train_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform)
val_ds = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(train_ds.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

model.eval()
correct = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item()
              for x, y in val_loader)
print(f"Accuracy: {100 * correct / len(val_ds):.2f}%")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), f"models/{args.model_out}")
