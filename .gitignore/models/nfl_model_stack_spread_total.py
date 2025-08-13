import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import random
import models.nfl_model_spread_total_gradient_boost as nfl_model_spread_total_gradient_boost
import nfl_model_spread_total_linear_reg
import requests
from bs4 import BeautifulSoup

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MetaModelRegressor(nn.Module):
    def __init__(self):
        super(MetaModelRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_meta_model(X_tensor, y_tensor):
    model = MetaModelRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/200 - Loss: {loss.item():.4f}")

    return model

def get_models():
    # Calling the nfl_model_spread_total_gradient_boost to get the outcome from the gradient boost model
    #gb_output_df = nfl_model_spread_total_gradient_boost.main()

    # Extract just the probabilities (make sure order matches your logistic regression predictions)
    # gb_spread_pred = gb_output_df['Pred_Spread'].values
    # gb_total_pred = gb_output_df['Pred_Total'].values

    # Calling the nfl_model_spread_total_linear_reg to get the outcome from the linear regression model
    lr_output_df = nfl_model_spread_total_linear_reg.main()

    # Extract just the probabilities
    lr_spread_pred = lr_output_df['PredictedSpread'].values
    lr_total_pred = lr_output_df['PredictedTotal'].values

    # Suppose your Logistic Regression output is saved as CSV with columns: Home, Visitor, Probability
    pinnacle_probs_df = pd.read_csv("csv_folder/Pinnacle_odds.csv")

    # Extract just the probabilities
    p_spread_pred = pinnacle_probs_df['VegasSpread'].values
    p_total_pred = pinnacle_probs_df['VegasTotal'].values

    # Stack model predictions as input features
    X_spread_stack = np.column_stack((lr_spread_pred))
    X_total_stack = np.column_stack((lr_total_pred))

    # Use Pinnacle data as soft targets
    y_spread = p_spread_pred.astype(np.float32)
    y_total = p_total_pred.astype(np.float32)

    # Convert to PyTorch tensors
    X_spread_tensor = torch.tensor(X_spread_stack, dtype=torch.float32)
    y_spread_tensor = torch.tensor(y_spread, dtype=torch.float32).unsqueeze(1)

    X_total_tensor = torch.tensor(X_total_stack, dtype=torch.float32)
    y_total_tensor = torch.tensor(y_total, dtype=torch.float32).unsqueeze(1)


    spread_model = train_meta_model(X_spread_tensor, y_spread_tensor)
    total_model = train_meta_model(X_total_tensor, y_total_tensor)

    # Put models in eval mode and make predictions
    spread_model.eval()
    total_model.eval()

    pred_spread = spread_model(X_spread_tensor).detach().numpy().flatten()
    pred_total = total_model(X_total_tensor).detach().numpy().flatten()

    # Save or print results
    output_df = pd.DataFrame({
        "Home": lr_output_df["Home"],
        "Visitor": lr_output_df["Visitor"],
        "StackedSpread": pred_spread,
        "StackedTotal": pred_total,
        "VegasSpread": pinnacle_probs_df['VegasSpread'].values,
        "VegasTotal": pinnacle_probs_df['VegasTotal'].values,
    })
    return output_df


def main():
    output_df = get_models()
    output_df.to_csv("csv_folder/week1_predictions/week1_stacked_spread_total_predictions.csv", index=False)

if __name__ == "__main__":
    main()



