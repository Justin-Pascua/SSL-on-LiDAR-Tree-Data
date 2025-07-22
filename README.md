# Overview
This is a semi-supervised learning project which makes use of LiDAR tree data to perform tree species classification. The dataset can be found here: https://zenodo.org/records/3549595

We begin by training an MLP on the labeled dataset. Noteably, within the labeled dataset, we observe extreme class imbalance (the top 3 classes make up about 75% of the labeled samples). To address this, we apply SMOTE to mitigate this imbalance, and use a weighted version of the cross entropy loss function when training the model. Doing so, we achieve a macro-F1 score of 0.781 on a test set taken from the labeled dataset (although the model still struggles with the minority classes).

Then, we use this trained MLP to generate pseudo-labels on the unlabeled dataset. We also assign probabilities, or confidence values, to each pseudo-label using the softmax function. We then form a new training set by taking the pseudo-labeled samples whose pseudo-label confidence is $\geq 0.75$. After training the model on this pseudo-labeled data, we achieve a macro-F1 score of 0.758 (on the same test set used before), which is marginally lower than the macro-F1 score achieved when training on just the labeled data.

# The Repository
The `src` folder contains the source code, which contains functions and classes used for preprocessing, data-splitting, model construction/training, and metric evaluation.

The `weights` folder contains pretrained weights. The `weights_trained_on_real.pth` file contains the weights of the model after it was trained on the labeled dataset. The `weights_trained_on_pseudo.pth` file contains the weights of the model after it was trained on a dataset of pseudo-labeled samples with sufficiently high confidence values (`threshold = 0.75`).

The `demo` folder contains a `.ipynb` folder demonstrating how the user can download the dataset and how to make use of the source code.
