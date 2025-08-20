import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def data_loader(df):
    x = np.stack(df['time_trace'])

    y = df['dis_to_target'].values

    # 4 classes: 0, 1, 2, 3, 4, 5
    n_bins = 6
    bins = np.linspace(y.min(), y.max(), n_bins + 1)
    y_class = np.digitize(y, bins[1:-1])  # produces class labels 0,1,2,3,4,5

    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y_class, test_size=0.2, random_state=42)

    # Scale only the input features
    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train)
    x_test_s = x_scaler.transform(x_test)

    # Do NOT scale y for classification!
    train_ds = TensorDataset(torch.tensor(x_train_s, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(x_test_s, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    return train_loader, test_loader

class CNNClassifier(nn.Module):
    def __init__(self, n_timepoints, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),   
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),   
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),  
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        # Calculate the output size after 3 pools (divide by 8, integer division)
        linear_input_size = 128 * (n_timepoints // 8)
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 256),  # Wider FC layer
            nn.ReLU(),
            nn.Dropout(0.3),                    # Regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)    # [batch, 1, n_timepoints]
        x = self.conv(x)
        return self.fc(x)
    
def evaluate(model, data_loader, loss_fn):
    model.eval()
    total, correct, running_loss = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            running_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    avg_loss = running_loss / total
    acc = 100 * correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return avg_loss, acc, all_preds, all_labels

def fit(model, train_loader, test_loader, epochs=20):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        avg_loss = running_loss / total
        acc = 100 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(acc)

        # Evaluate on test
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, loss_fn)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs} Train Loss: {avg_loss:.4f}  Train Accuracy: {acc:.2f}%  Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.2f}%')
        
    torch.save({
        'model_state_dict': model.state_dict()
    }, "C:/Users/vpming/tuni_ml/src/model/balanced_cnn_classifier__cmax_cvar.pt")

    # Plot losses and accuracies
    epochs_range = np.arange(1, epochs+1)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')

    # Confusion matrix for final test predictions
    plt.subplot(1, 3, 3)
    conf_mat = confusion_matrix(test_labels, test_preds)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()

    return train_losses, test_losses, train_accs, test_accs, conf_mat

def train_per_noise(df):
    results = []

    for noise_level in sorted(df['noise'].unique()):
        print(f'\n=== Training for noise {noise_level} ===')
        mask = df['noise'] == noise_level
        X = np.stack(df.loc[mask, 'time_trace'])
        y = df.loc[mask, 'dis_to_target'].values  

        n_bins = 6
        bins = np.linspace(y.min(), y.max(), n_bins + 1)
        y_class = np.digitize(y, bins[1:-1])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32),
                                torch.tensor(y_train, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test_s, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        # Define model for this run
        n_timepoints = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        model = CNNClassifier(n_timepoints, n_classes)

        # Train & evaluate (use your fit function)
        print(f"Training on {X_train.shape[0]} samples; Testing on {X_test.shape[0]} samples")
        train_losses, test_losses, train_accs, test_accs, conf_mat = fit(model, train_loader, test_loader, epochs=15)
        results.append({'noise': noise_level, 'test_acc': test_accs[-1], 'confusion_matrix': conf_mat})

    # Plot performance vs. noise
    plt.plot([r['noise'] for r in results], [r['test_acc'] for r in results], marker='o')
    plt.xlabel("Noise Level")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title("Model Accuracy vs. Noise") 
    plt.show()