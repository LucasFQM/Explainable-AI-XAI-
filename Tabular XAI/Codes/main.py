
import random
import numpy as np
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch import nn

from data import Data
from model import NeuralNetwork
from training import Trainer
from plots import LossPlotter
from XAI.PFI import PermutationFeatureImportance
from XAI.Others.DCI import DropColumnImportance


# ------------------ Reproducibility --------------------------------

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.use_deterministic_algorithms(True)


# ------------------ Load Dataset -----------------------------------

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
feature_names = X.columns.tolist()


# ------------------ Split Dataset ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=seed,
    stratify=y
)


# ------------------ Standardize Features --------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------ DataLoader -------------------------------------

batch_size = 32

train_data = Data(X_train, y_train)
test_data = Data(X_test, y_test)

generator = torch.Generator()
generator.manual_seed(seed)

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    generator=generator
)


# ------------------ Define Model -----------------------------------

input_dim = X_train.shape[1]
hidden_dim = 10
output_dim = 1

model = NeuralNetwork(input_dim, hidden_dim, output_dim)


# ------------------ Loss Function and Optimizer -------------------

learning_rate = 0.01
epochs = 20

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)


# ------------------ Train Model ------------------------------------

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer
)

loss_values = trainer.train(num_epochs=epochs)


# ------------------ Plot Training Loss -----------------------------

# loss_plotter = LossPlotter(loss_values)
# loss_plotter.plot_loss()


# ------------------ Permutation Feature Importance (PFI) -----------

pfi = PermutationFeatureImportance(
    model=model,
    X_test=test_data.X,
    y_test=test_data.y,
    feature_names=feature_names,
    n_repeats=30,
    random_state=seed
)

baseline_accuracy, importances = pfi.calculate_importance()

#print(f"Baseline Accuracy: {baseline_accuracy:.4f}")

# Plot Feature Importance
#pfi.plot_importance(importances)


# ------------------ End of PFI -------------------------------------


# ------------------ Drop Column Importance (DCI) ----------------------


dci = DropColumnImportance(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    hidden_dim=hidden_dim,
    learning_rate=learning_rate,
    epochs=epochs,
    batch_size=batch_size,
    random_state=42
)

baseline_accuracy_dci, dci_importances = dci.calculate_importance()

print(f"\nDCI Baseline Accuracy: {baseline_accuracy_dci:.4f}")

dci.plot_importance(dci_importances)

# ------------------ End of DCI -------------------------------------