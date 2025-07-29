# 🏈 NFL Match Predictor

This project uses historical NFL play-by-play and box score data to build a logistic regression model that predicts the probability of a home team winning upcoming matchups. The model is deployed using a Flask web app.

---

## 📊 Features

- Cleans and combines multiple seasons of play-by-play and box score data
- Engineers features like:
  - Win rate
  - Points scored and allowed
  - Yardage metrics
  - Completion rates
  - Turnover rates
- Builds a logistic regression model with cross-validation
- Predicts the probability of a home win for 2025 Week 1 games
- Exposes results through a Flask web app and `week1_predictions.csv`

---

## 🧠 Model

The model is trained using data from the 2023 and 2024 NFL seasons.

- **Algorithm**: Logistic Regression
- **Target**: Binary — whether the home team won
- **Features**: Engineered differences in team performance metrics
- **Cross-Validation**: 5-fold, average accuracy printed on training

---

## 🛠️ Technologies

- Python 3
- Flask
- Pandas
- Scikit-learn
- HTML/CSS/Jinja2 (for frontend)

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/nfl-matchup-predictor.git
cd nfl-matchup-predictor
```
