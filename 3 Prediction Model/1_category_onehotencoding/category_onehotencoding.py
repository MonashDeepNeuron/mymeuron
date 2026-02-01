# import necessaries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

# from tqdm import tqdm


# setting fixed seed "42"
def set_seed(seed: int = 42):
    """
    Fix seed at 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Learned from UOM course:
class EarlyStopping:
    """
    Early stopping class, learned from UOM MMC courses.

    Args:
        patience (int): How many epochs to wait for a metric to improve.
        minimise (bool): Whether the metric we are monitoring is to minimise,
        such as loss, or maximise,
        such as accuracy.
    """

    def __init__(self, patience: int = 5, minimise: bool = True) -> None:
        self.limit = patience
        self.minimise = minimise
        self.init()

    def improved(self) -> bool:
        if self.minimise:
            return self.last_ < self.best_
        else:
            return self.last_ > self.best_

    def init(self):
        self.count_: int = 0
        self.best_: float = float("inf" if self.minimise else "-inf")
        self.last_: float = None

    def check(self, new: float) -> int:
        """
        Checks a new value of a metric and return a status code
        to indicate whether it has improved since the last check.

        Args:
            new (float): The new value for the metric to check.

        Returns: Status code, either
            0-Improved,
            1-Not improve (patience),
            2-Not improve (stop).
        """
        self.last_ = new
        if self.improved():
            self.best_ = new
            self.count_ = 0
            return 0
        self.count_ += 1

        if self.count_ <= self.limit:
            print(
                f"  [EarlyStopping] Patience {self.count_}/{self.limit}, best={self.best_:.4f}"
            )
            return 1
        else:
            return 2


# Have to discuss chose which model to train and triage for "Regression" data.
# Choice 1: Linear Regression Skit learn


# Choice 2:
class MultiLayerPerceptron(nn.Module):
    "A simple and configurable multi-layer perceptron."

    def __init__(
        self,
        in_features: int,  # inputs attributes
        layer_units: list[int],  # hidden layer
    ):
        """Constructor for MLP.

        Args:
            in_features: Number of input features.
            n_classes: Number of output classes.
            layer_units: List of units in each hidden layer.
        """
        super().__init__()

        # Create the layers
        layers: list[nn.Module] = []
        prev_units: int = in_features  # previous layer

        # Add hidden layers loop
        for units in layer_units:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.GELU())  # GELU > RELU, activate function
            prev_units = units  # update prev_units for next loop

        # Add output layer
        layers.append(
            nn.Linear(prev_units, 1)
        )  # add last hidden layer to output "n_class"

        # Create the sequential model
        self._hidden_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Input -> Hidden1 linear -> GELU -> Hidden2 linear -> Output(1)
        return self._hidden_layers(x)


# use in preprpocessing tackle with outlier
"""
def handle_outliers(df, column_name, method="clip"):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    IQR = q3 - q1

    lower_bound = q1 - (1.5 * IQR)
    upper_bound = q3 + (1.5 * IQR)

    # run clip restrict range from lower to upper
    df[column_name] = df[column_name].clip(lower=lower_bound, upper=upper_bound)
    return df
"""


# can change csv file name anytime
def col_transform(csv_path: str = "paintrainingdata.csv") -> pd.DataFrame:
    """
    focus in column transform:
    category columns "gender", "race", "arrival_transport" using one-hot encoding process
    """
    df = pd.read_csv(csv_path)
    df = df.copy()

    # define category columns
    categories_columns = [
        "gender",
        "race",
        "arrival_transport"
    ]
    # using one-hot encoding categories_columns
    #into ex: gender_m, gender_f, gender_x, race_white, race_black.....many columns
    df = pd.get_dummies(df, columns = categories_columns, dtype = int)

    #df = df.dropna()

    return df


def load_data(df: pd.DataFrame):
    """
    columns:
        x = [
        "age",
        "gender",
        "race",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",  # X: vitals
        "pain",
        # 'chiefcomplaint', # too hard from now
        ]

        y = "Level of Care"  # y: Target
    """

    # X:numeric_features:
    numeric_features = [
        "age",
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
    ]
    # X:categories_features:
    exclude_columns = numeric_features + ["Level of Care", "subject_id", "stay_id", "intime", "chiefcomplaint", "Unnamed: 0"]
    categories_features = [col for col in df.columns if col not in exclude_columns]

    # all x features with order: numeric_features + categories_features:
    final_x_features = numeric_features + categories_features

    X = df[final_x_features].values.astype(np.float32)
    y = df["Level of Care"].values.astype(np.float32)

    """
    # Future: can use kfold, testing baseline performance
    cross_validator = KFold(n_splits=5, shuffle=True, random_state=42) # 5-folds
    model = LinearRegression(max_iter=1000)

    # sklearn evaluate, scores is numpy array
    scores = cross_val_score(
        estimator = model,
        X = X,
        y = y,
        cv = cross_validator,)
    """

    # Split Data: Train/Val/Test to 60:20:20
    # first split: (train 80/ test 20):    8:2
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # second split from 0.8: 0.75:0.25 -> train 60, val 20, test 20
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )


    # Standardization
    standard_scaler = StandardScaler()
    # choose only numeric_features into fit scaler:
    numeric_count = len(numeric_features)
    X_train[:, :numeric_count] = standard_scaler.fit_transform(X_train[:, :numeric_count])
    X_val[:, :numeric_count] = standard_scaler.transform(X_val[:, :numeric_count])
    X_test[:, :numeric_count] = standard_scaler.transform(X_test[:, :numeric_count])

    # Convert to PyTorch tensors
    # train
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    # val
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    # test
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    # ******************* have to change batch_size when read large data csvs***********************************************
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

    n_features = X_train.shape[1] #update total features

    return train_loader, val_loader, test_loader, n_features


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_features: int,
    layer_units: list[int],
    num_epochs: int,
    learning_rate: float,
):
    """
    Train MLP regression model
    """

    model = MultiLayerPerceptron(in_features=n_features, layer_units=layer_units)

    # regression using nn.MSEloss mean squred error
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    earlystop_tracker = EarlyStopping(patience=5, minimise=True)
    best_model_saved = False

    # Training loop: to find the minimum (global minimum) of the loss
    for epoch in range(num_epochs):
        # ================= Training ========================
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)  # forward
            train_loss = loss_fn(outputs, y_batch)  # loss function

            # Backward pass and optimization
            optimizer.zero_grad()  # to zero
            train_loss.backward()  # back propagation
            optimizer.step()  # update

            train_losses.append(train_loss.item())
        avg_train_loss = float(np.mean(train_losses))

        # ================= Validation ========================
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss = loss_fn(outputs, y_batch)
                val_losses.append(val_loss.item())
        avg_val_loss = np.mean(val_losses)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train MSE: {avg_train_loss:.4f} | "
            f"Val MSE: {avg_val_loss:.4f}"
        )

        # ================= Check early stop ========================
        earlystop_status = earlystop_tracker.check(avg_val_loss)
        if earlystop_status == 0:  # Improves
            torch.save(model.state_dict(), "mlp_regression_model.pth") # move the statement outside the for loop and before the return statement
            best_model_saved = True
        if earlystop_status == 2:  # Did not improve
            print("Early stopping triggered.")
            break

    if best_model_saved:
        model.load_state_dict(torch.load("mlp_regression_model.pth"))

    return model


def evaluate(model, test_loader):
    """
    Evaluate regression performance on test set.
    """

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            y_true.extend(y_batch.squeeze(1).numpy())
            y_pred.extend(output.squeeze(1).numpy())

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    print("\nTest set performance:")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R^2 : {r2:.4f}")


def main() -> None:
    set_seed(42)  # fixed seed

    df = col_transform("paintrainingdata.csv")
    # print(df.head(10))

    train_loader, val_loader, test_loader, n_features = load_data(df)
    layer_units = [32, 64, 32]
    model = train(
        train_loader,
        val_loader,
        n_features,
        layer_units,
        num_epochs=100, # increased epochs, because I have "Early stop"
        learning_rate=0.001,
    )

    evaluate(model, test_loader)


if __name__ == "__main__":
    main()
