import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import random

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Suppose your XGBoost output is saved as CSV with columns: Home, Visitor, Probability
xgb_output_df = pd.read_csv("csv_folder/week1_predictions_xgboost.csv")

# Extract just the probabilities (make sure order matches your logistic regression predictions)
xgb_probs = xgb_output_df['HomeWinProbability'].values

# Suppose your Logistic Regression output is saved as CSV with columns: Home, Visitor, Probability
lr_output_df = pd.read_csv("csv_folder/week1_predictions_logistic_reg.csv")

# Extract just the probabilities
lr_probs = lr_output_df['HomeWinProbability'].values

# Suppose your Logistic Regression output is saved as CSV with columns: Home, Visitor, Probability
pinnacle_probs_df = pd.read_csv("csv_folder/Pinnacle_Probs.csv")

# Extract just the probabilities
p_probs = pinnacle_probs_df['HomeWinProbability'].values

xgb_probs = xgb_probs.astype(float)
lr_probs = lr_probs.astype(float)
p_probs = p_probs.astype(float)


X_stack = np.column_stack((xgb_probs, lr_probs))

# Target is Pinnacle's probabilities (soft labels)
y_soft = p_probs


class MetaModelNN(nn.Module):
    def __init__(self):
        super(MetaModelNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),   # Inputs: xgb_prob, lr_prob
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
    


X_tensor = torch.tensor(X_stack, dtype=torch.float32)
y_tensor = torch.tensor(y_soft.reshape(-1, 1), dtype=torch.float32)


model = MetaModelNN()
criterion = nn.MSELoss()  # Or nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")



model.eval()
with torch.no_grad():
    combined_preds = model(X_tensor).numpy().flatten()

combined_pred_df = pd.DataFrame({
    "Home": xgb_output_df['Home'].values,
    "Visitor": xgb_output_df['Visitor'].values,
    "HomeWinProbability": combined_preds
})
print(combined_pred_df)