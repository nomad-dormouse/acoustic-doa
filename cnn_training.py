import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from dataset_preprocessing import N_MELS

class TinyCNN(nn.Module):
    def __init__(self, in_mels=N_MELS, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear((in_mels//2) * (int(np.ceil((16000*1.0)/256)//2)) * 16, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x shape: [B, 1, Mels, Time]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # reduce again
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_processed_data(train_path="./data/dads/train.json", test_path="./data/dads/test.json"):
    """Load processed data from JSON"""
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    return train_data, test_data

def evaluate_model(model, test_data):
    """Evaluate model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sample in test_data:
            mel_spec = torch.tensor(sample['mel_spectrogram'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            label = torch.tensor(sample['label'], dtype=torch.long)
            
            output = model(mel_spec)
            _, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted == label).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def train_model(model, train_data, test_data, num_epochs=5):
    """Train the CNN model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for sample in train_data:
            mel_spec = torch.tensor(sample['mel_spectrogram'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            label = torch.tensor(sample['label'], dtype=torch.long)
            
            optimizer.zero_grad()
            output = model(mel_spec)
            loss = criterion(output, label.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate on test set after training
    evaluate_model(model, test_data)
    
    return model

if __name__ == "__main__":
    train_data, test_data = load_processed_data()
    model = TinyCNN()
    trained_model = train_model(model, train_data, test_data)
    torch.save(trained_model.state_dict(), "./data/models/trained_cnn.pth")
