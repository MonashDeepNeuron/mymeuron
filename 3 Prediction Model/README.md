# 20251225 Eddy update mymeuron project
This folder contains an initial baseline first experiment py for training "paintrainingdata.csv", by using a "MLP Mult-Layer Perceptron regression model".

# 20250201 Eddy update:
- This python file updated categories features (such as "gender", "race", "arrival_transport")using "One-Hot Encoding" method.
- Not letting "categories_features" into standardscaler to transform.
- debug "Early stop" in train().

# Purpose
This change only contains "numeric" and "category" input features.
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

categories_features = [
    "gender",
    "race",
    "arrival_transport"
]
# But
***** X: NOT INCLUDE: "chiefcomplaint" right now, text data need further development, still need some studies.


# Model explaining
Method: LinearRegressionModel 
Model: MLP
Loss function: nn.MSELoss()
Optimizer: optim.Adam
Extra function: Early stop applied, 20260201 debugged, now saved the best model.

# Current Status
Result of "Test set" performance:

20251226:
MSE  = 795.58
RMSE = 28.21
R^2  = -0.5055

20260201:
MSE : 471.1089
RMSE: 21.7050
R^2 : 0.1085

My conclusion:
After using "One-Hot Encoding" for categories_features, and not letting categories_features to standard_scaler, the result is a lot better than the initial version.

# Future development plans:
- start tackle text data into vector ("chiefcomplaint")
- "chiefcomplaint" keywords expand. (not only "pain")
- improve the "preloc_preprocessing.py", and redo preprocessing using coding method to calculate "Average_LOC".

# Notes:
- This is *not a final model or a workable model*.
- Features set will be expanded in future versions (chiefcomplaint)

# Developer
Eddy



