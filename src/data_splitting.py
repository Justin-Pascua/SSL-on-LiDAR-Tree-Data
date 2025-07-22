import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

class TreeDataset(Dataset):
    def __init__(self, data_array, transform=None, target_transform=None):
        self.X = torch.from_numpy(data_array[:, :-1].copy()).to(device)
        self.y = torch.from_numpy(data_array[:, -1:].copy()).to(device)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        y = self.y[i]
        if self.target_transform:
          y = self.target_transform(self.y[i])
        return x.to(device), y.to(device)

# Define one-hot-encoding transformation
def one_hot(y):
  """
  Converts integers in {0, 1, ..., 7} to its one-hot encoding
  params:
    y: a singleton tensor whose only entry is an integer in {0, 1, ..., 7}
  """
  
  y = int(y.item())
  return torch.zeros(8).scatter_(0, torch.tensor(y), value = 1).to(device)

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, feature_columns):
        self.data = dataframe[feature_columns].values
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32).to(device)

def predict_on_dataframe(model, dataframe, feature_columns, batch_size = 1024):
    """
    Applies the given model to a dataframe of unlabeled tree data
    to generate pseudo-labels.
    params:
        model: a PyTorch model compatible with the LiDAR Tree data
        
        dataframe: a pandas dataframe containing the unlabeled data samples
        
        feature_columns: a list of columns in dataframe which represent the features
        
        batch_size: an integer indicating the batch size used for the dataloaders applied
        to the dataframe
    """
    model = model.to(device)
    model.eval()

    dataset = DataFrameDataset(dataframe, feature_columns)
    loader = DataLoader(dataset, batch_size = batch_size)

    predictions = []

    probabilities = []
    predictions = []

    with torch.no_grad():
      for batch in loader:
        outputs = nn.Softmax(dim = 1)(model(batch))
        probs, preds = torch.max(outputs, 1)

        probabilities.append(probs.cpu().numpy())
        predictions.append(preds.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(probabilities)

def get_labeled_datasets(full_df, test_size = 0.5, val_size = 0.2, 
                         smote = False, smote_scaling_dict = None):
    """
    Given a pandas dataframe of labeled data samples, this method returns 
    PyTorch datasets for training, validation, and testing, as well as the full dataset
    params:
        full_df: a pandas dataframe containing the labeled data samples
        
        test_size: a number in [0, 1] indicating what portion of the full 
        dataset will be used for testing
        
        val_size: a number in [0, 1] indicating (after splitting the test data 
        from the rest of the data set) what portion of the training data will 
        be used for validation
        
        smote: a boolean indicating whether to use SMOTE to generate samples 
        for minority classes
        
        smote_scaling_dict: a dictionary of the form 
        {[CLASS NUMBER]: [SCALAR APPLIED TO ORIGINAL NUMBER OF SAMPLES OF CLASS i]}
    """
    
    # split full_df into train/val/test
    train_df, test_df = train_test_split(full_df, test_size = test_size)
    train_df, val_df = train_test_split(train_df, test_size = val_size)
    
    # if specified, generate samples using SMOTE
    if smote:
        counts = dict(train_df['Species Number'].value_counts())
        desired_num_samples = {i: int(counts[i] * smote_scaling_dict[i]) for i in range(8)}
        X_train = train_df.drop(columns = ['Species', 'Species Number'])
        y_train = train_df['Species Number']
        
        smote = SMOTE(sampling_strategy = {i: desired_num_samples[i] for i in range(8)})
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        train_smote_df = pd.concat([X_smote, y_smote], axis = 1)
        
    # convert into numpy arrays
    train_np = np.array([])
    if smote:
        train_np = train_smote_df.to_numpy(dtype = np.float32)
    else:
        train_np = train_df.drop(['Species'], axis = 1).to_numpy(dtype = np.float32)
        
    val_np = val_df.drop(['Species'], axis = 1).to_numpy(dtype = np.float32)
    test_np = test_df.drop(['Species'], axis = 1).to_numpy(dtype = np.float32)
    full_np = full_df.drop(['Species'], axis = 1).to_numpy(dtype = np.float32)

    # convert into PyTorch datasets
    train_dataset = TreeDataset(train_np, target_transform = one_hot)
    val_dataset = TreeDataset(val_np, target_transform = one_hot)
    test_dataset = TreeDataset(test_np, target_transform = one_hot)
    full_dataset = TreeDataset(full_np, target_transform = one_hot)
    
    return train_dataset, val_dataset, test_dataset, full_dataset

def get_pseudo_labeled_dataframe(model, full_unlabeled_df):
    """
    Applies the given model to generate pseudo-labels and prediction confidence 
    values on a dataframe of unlabeled samples. 
    Note: the columns of the unlabeled dataframe must all be features. 
    If a threshold is specified, then predictions with confidence lower than the 
    threshold are omitted and filled with the value -1
    params:
        model:a PyTorch model compatible with the LiDAR Tree data
        
        full_pseudo_labeled_df: a pandas dataframe containing the unlabeled 
        data sample
        
        threshold: a number in [0, 1] indicating how confident a pseudo-label 
        must be to be kept in the dataset
    """
    # get list of features
    features = list(full_unlabeled_df.columns)
    
    # get pseudo-labels and probabilities
    predictions, probabilities = predict_on_dataframe(model, full_unlabeled_df, features)

    # store pseudo-labels and probabilities in dataframe    
    full_pseudo_labeled_df = full_unlabeled_df.copy()
    full_pseudo_labeled_df['Predicted Species Number'] = predictions
    full_pseudo_labeled_df['Prediction Confidence'] = probabilities
    
    return full_pseudo_labeled_df

def get_pseudo_labeled_datasets(model, full_pseudo_labeled_df, threshold = 0.75, 
                                smote = False, smote_scaling_dict = None):
    """
    Given a pandas dataframe of pseudo-labeled data samples, returns a PyTorch 
    dataset consisting of samples whose prediction confidence exceeds the 
    specified threshold.
    params:
        model: a PyTorch model compatible with the LiDAR Tree data
        
        full_pseudo_labeled_df: a pandas dataframe containing the 
        pseudo-labeled data samples and prediction confidence values
        
        threshold: a number in [0, 1] indicating how confident a pseudo-label 
        must be to be kept in the dataset
        
        smote: a boolean indicating whether to use SMOTE to generate samples 
        for minority classes
        
        smote_scaling_dict: a dictionary of the form 
        {[CLASS NUMBER]: [SCALAR APPLIED TO ORIGINAL NUMBER OF SAMPLES OF CLASS i]}
    """
    # omit rows with low confidence
    high_confidence_mask = full_pseudo_labeled_df['Prediction Confidence'] > threshold
    full_pseudo_labeled_df['Predicted Species Number'] = full_pseudo_labeled_df['Predicted Species Number'].where(high_confidence_mask, -1)
    confident_pseudo_labeled_df = full_pseudo_labeled_df[full_pseudo_labeled_df['Prediction Confidence'] >= threshold]
       
    # if specified, generate samples using SMOTE
    if smote:
        counts = dict(confident_pseudo_labeled_df['Predicted Species Number'].value_counts())
        desired_num_samples = {i: int(counts[i] * smote_scaling_dict[i]) for i in range(8)}
        X_train = confident_pseudo_labeled_df.drop(columns = ['Predicted Species Number', 'Prediction Confidence'])
        y_train = confident_pseudo_labeled_df['Predicted Species Number']
        
        smote = SMOTE(sampling_strategy = {i: desired_num_samples[i] for i in range(8)})
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        train_smote_df = pd.concat([X_smote, y_smote], axis = 1)
        
    # convert into numpy arrays
    train_np = np.array([])
    if smote:
        train_np = train_smote_df.to_numpy(dtype = np.float32)
    else:
        train_np = confident_pseudo_labeled_df.drop(['Prediction Confidence'], axis = 1).to_numpy(dtype = np.float32)

    # convert into PyTorch datasets
    train_dataset = TreeDataset(train_np, target_transform = one_hot)
    
    return train_dataset





