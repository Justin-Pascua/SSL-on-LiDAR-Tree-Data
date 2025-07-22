import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import numpy as np

species_strings = ['Norway Spruce', 'European Larch', 'Other broadleaves', 
                   'Silver fir', 'Broadleaves', 'Green alder', 
                   'Pines', 'Scots Pine']

def one_hot_decode(y):
  return y.argmax()

def get_confusion_matrix(model, labeled_dataloader, normalize = 'true'):
  y_true, y_pred = [], []
  with torch.no_grad():
      for x,y in labeled_dataloader:
          preds = model(x).argmax(1)
          y_true.extend(y.argmax(1).cpu().numpy())
          y_pred.extend(preds.cpu().numpy())

  cm = confusion_matrix(y_true, y_pred, normalize = normalize)
  return cm

def plot_confusion_matrix(model, labeled_dataloader, normalize = 'true'):
  cm = get_confusion_matrix(model, labeled_dataloader, normalize = normalize)
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, cmap = plt.cm.Oranges,
              xticklabels = species_strings,
              yticklabels = species_strings)
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.title(f"Confusion Matrix Normalized By '{normalize}'")
  plt.show()
  
def get_f1_scores(model, labeled_dataloader):
  cm = get_confusion_matrix(model, labeled_dataloader, normalize = None)
  n_classes = cm.shape[0]
  f1_scores = []

  for i in range(n_classes):
    # True Positives (TP)
    tp = cm[i, i]

    # False Positives (FP) = Sum of column i (excluding TP)
    fp = np.sum(cm[:, i]) - tp

    # False Negatives (FN) = Sum of row i (excluding TP)
    fn = np.sum(cm[i, :]) - tp

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

  return np.array(f1_scores)

def print_f1_scores(model, labeled_dataloader):
    f1_scores = get_f1_scores(model, labeled_dataloader)
    for i, species in enumerate(species_strings):
        print(f"{species}: {f1_scores[i]: .3f}")
    print(f"Macro F1 Score: {np.mean(f1_scores): .3f}")
  
    
def plot_f1_scores(model, labeled_dataloader):
  raw_f1_scores = get_f1_scores(model, labeled_dataloader)
  macro_f1 = np.mean(raw_f1_scores)
  variance = np.var(raw_f1_scores)

  plt.bar(species_strings, raw_f1_scores, label = f'Variance = {variance: .3f}')
  plt.axhline(y = macro_f1, color='r', linestyle='--', label = f'Macro-F1 = {macro_f1: .3f}')
  plt.xticks(rotation = 45)
  plt.ylabel('F1 Score')
  plt.legend()
  plt.show()