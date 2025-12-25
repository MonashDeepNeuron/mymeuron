# 20251225 Eddy update mymeuron project

This folder contains an initial baseline first experiment py for training "paintrainingdata.csv", by using a "MLP Mult-Layer Perceptron regression model".

# Purpose
This first experiment only contains numeric "input features"
- Demographics: age, gender
- Vital signs (temperature, heart rate, respiratory rate, SpO₂, SBP, DBP)
- Pain score

# But
***** X: NOT CONTAINING: "race", "chiefcomplaint"
text data need further development.

# Model explaining
Method: LinearRegressionModel 
Model: MLP
Loss function: nn.MSELoss()
Optimizer: optim.Adam
Extra function: Early stop applied

# Current Status
Result of "Test set" performance:
MSE  = 795.58
RMSE = 28.21
R^2  = -0.5055

My conclusion:
This is a terrible model by far, not for working
Although "Training loss" and "Validation loss" decrease a lot,
RMSE = 28, means "Huge error" in prediction (+-28)
R^2 is negative means this model is useless, a raw prediction or guess would better than this, and input features are insufficient to predict Level of care.

# Future development plans:
- redo preprocessing, even numeric and text data
- start tackle text data into vector ("race", "chiefcomplaint")
- "chiefcomplaint" keywords expand. (not only "pain")

# Notes:
- This is *not a final model or a workable model*.
- Features set will be expanded in future versions (race, chiefcomplaint)

# Developer
Eddy



