import pandas as pd
import numpy as np
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from matplotlib import pyplot as plt

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class TreeClassifier(nn.Module):
    def __init__(self):
        """
        Initializes the model
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(15, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
            )
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, x):
        """
        Applies the model to a data sample x.
        params:
            x: a PyTorch tensor of shape (15,)
        """
        return self.linear_relu_stack(x)

    def predict(self, x):
        """
        Generates a prediction by calling self.forward and applying the softmax function
        params:
            x: a PyTorch tensor of shape (15,)
        """
        logits = self.forward(x)
        probabilities = nn.Softmax(dim = 0)(logits)
        return torch.argmax(probabilities).item()

    def _train_loop(self, train_dataloader, loss_fn, optimizer):
        """
        Performs a single training loop.
        params:
            train_dataloader: a torch.utils.data DataLoader for the labeled training data
            
            loss_fn: a torch.nn loss function
            
            optimizer: a torch.optim optimizer
        """
        self.train()
        for batch, (X,y) in enumerate(train_dataloader):
            pred = self(X)
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def _training_eval(self, train_dataloader, val_dataloader, loss_fn, verbose = 1):
        """
        Evaluates model accuracy and loss value on the trianing set and validation set
        using current model weights. Returns these values in the following order: train accuracy, train loss, validation accuracy, validation loss
        params:
            train_dataloader: a torch.utils.data DataLoader for the labeled training data
            
            val_dataloader: a torch.utils.data DataLoader for the labeled validation data
            
            loss_fn: a torch.nn loss function
            
            verbose: an integer (either 0 or 1) indicating whether or not to 
            print the accuracy and loss values. 
            If verbose == 0, then no print statement is executed.
            If verbose == 1, then this method prints the returned values
        """
        self.eval()
        
        train_size = len(train_dataloader.dataset)
        val_size = len(val_dataloader.dataset)
        
        num_train_batches = len(train_dataloader)
        num_val_batches = len(val_dataloader)
        
        train_loss, train_correct = 0, 0
        val_loss, val_correct = 0, 0
        
        with torch.no_grad():
            for X, y in train_dataloader:
                pred = self(X)
                train_loss += loss_fn(pred, y).item()
                train_correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            for X, y in val_dataloader:
                pred = self(X)
                val_loss += loss_fn(pred, y).item()
                val_correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        
        train_correct /= train_size
        train_loss /= num_train_batches
        
        val_correct /= val_size
        val_loss /= num_val_batches
        
        if verbose == 1:
            print(f"Train Accuracy: {(train_correct * 100):.3f}%, Avg loss: {train_loss:>8f}, Val Accuracy: {(val_correct * 100):.3f}%, Avg loss: {val_loss:>8f}")
        
        return train_correct*100, train_loss, val_correct*100, val_loss

    def fit(self, train_dataloader, val_dataloader, loss_fn, optimizer, epochs, verbose = 1):
        """
        Trains the model on dataset associated with train_dataloader.
        params:
            train_dataloader: a torch.utils.data DataLoader for the labeled training data
                      
            val_dataloader: a torch.utils.data DataLoader for the labeled validation data
                      
            loss_fn: a torch.nn loss function
                      
            optimizer: a torch.optim optimizer
                      
            epochs: integer representing number of times to iterate over the training data
                      
            verbose: an integer (either 0 or 1) indicating whether or not to 
            print the accuracy and loss values. 
            If verbose == 0, then no print statement is executed.
            If verbose == 1, then this method prints the returned values          
        """
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []
        
        for t in range(epochs):
            if verbose == 1:
                print(f"Epoch {t+1}")
            
            self._train_loop(train_dataloader, loss_fn, optimizer)
            
            train_acc, train_loss, val_acc, val_loss = self._training_eval(train_dataloader, val_dataloader, loss_fn, verbose)
            
            train_acc_history.append(train_acc)
            train_loss_history.append(train_loss)
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            
            if verbose == 1:
                print('\n-------------------------------')
        
        print("Done!")
        
        return (train_acc_history, train_loss_history, val_acc_history, val_loss_history)

    def evaluate_accuracy(self, labeled_dataloader, loss_fn):
        """
        Evaluates and prints model accuracy and loss value on a single dataset
        params:
            labeled_dataloader: a torch.utils.data DataLoader for labeled data
            
            loss_fn: a torch.nn loss function
        """
        self.eval()
        size = len(labeled_dataloader.dataset)
        num_batches = len(labeled_dataloader)
        loss, correct = 0, 0
        
        with torch.no_grad():
            for X, y in labeled_dataloader:
                pred = self(X)
                loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        
        correct /= size
        loss /= num_batches
        print(f"Accuracy: {(correct * 100):.3f}%, Avg loss: {loss:>8f}")
    
    
def plot_training_progress(history):
    """
    Plots the training history of the model. One plot visualizes model accuracy,
    and the other visualizes the loss function.
    params:
        history: a list of 4 numpy arrays, as returned by TreeClassifier.fit
    """
    train_acc_history, train_loss_history, val_acc_history, val_loss_history = history
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 5))
    
    epochs = len(history[0])
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].plot([n for n in range(epochs)], train_acc_history,
                 color = 'blue', label = 'Training Accuracy')
    axes[0].plot([n for n in range(epochs)], val_acc_history,
                 color = 'orange', label = 'Validation Accuracy')
    axes[0].legend()
    axes[0].set_ylim(ymin = 0)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss')
    axes[1].plot([n for n in range(epochs)], train_loss_history,
                 color = 'blue', label = 'Training Loss')
    axes[1].plot([n for n in range(epochs)], val_loss_history,
                 color = 'orange', label = 'Validation Loss')
    axes[1].legend()
    axes[1].set_ylim(ymin = 0)
    
    plt.show()