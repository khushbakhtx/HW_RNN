import os
import logging
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train.log'),
        logging.StreamHandler()
    ]
)

logger.add('logs/train.log', rotation='1 MB', retention='10 days', level='INFO', format='{time} - {level} - {message}')

class FastTextDataset(Dataset):
    def __init__(self, X: list, y: np.ndarray, max_seq_length: float = 400):
        self.X = X
        self.y = y
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = torch.tensor(self.X[index]).float()
        y = torch.tensor(self.y[index]).float()
        seq_length = torch.tensor(self.X[index].shape[0])
        X = nn.functional.pad(X, (0, 0, 0, self.max_seq_length - X.shape[0]))
        return X, y, seq_length

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

def train_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, 
                train_dataloader: DataLoader, test_dataloader: DataLoader, num_epochs: int, device: str):
    
    train_loss_hist = []
    test_loss_hist = []
    
    for epoch in range(num_epochs):
        train_losses_epoch = []
        test_losses_epoch = []
        train_acc_epoch = []
        test_acc_epoch = []
        
        model.train()
        
        for X_batch, y_batch, text_lengths in tqdm(train_dataloader):
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output_train = model(X_batch, text_lengths).squeeze()
            loss_train = criterion(output_train, y_batch)
            loss_train.backward()
            optimizer.step()
            train_acc_epoch.append(binary_accuracy(output_train, y_batch).item())
            train_losses_epoch.append(loss_train.item())
        
        model.eval()
        
        with torch.no_grad():
            for X_batch_valid, y_batch_valid, val_text_lengths in test_dataloader:
                X_batch_valid, y_batch_valid = X_batch_valid.to(device), y_batch_valid.to(device)
                output_test = model(X_batch_valid, val_text_lengths).squeeze()
                loss_test = criterion(output_test, y_batch_valid)
                acc_test = binary_accuracy(output_test, y_batch_valid)
                test_losses_epoch.append(loss_test.item())
                test_acc_epoch.append(acc_test.item())
        
        train_loss = np.mean(train_losses_epoch)
        test_loss = np.mean(test_losses_epoch)
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)

        log_message = f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        print(log_message)
        logger.info(log_message)
        logging.info(log_message)

    return train_loss_hist, test_loss_hist
