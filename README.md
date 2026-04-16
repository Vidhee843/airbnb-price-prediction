# Airbnb Listing Price Prediction 🏠

A machine learning project that predicts Airbnb listing prices across 4 major US cities using an end-to-end ML pipeline built in Python.

## Overview
This project analyzes 5,618 Airbnb listings across **Boston, NYC, Chicago, and San Francisco** to identify key price drivers and build a predictive model for listing prices.

## Key Results
- 🏆 Random Forest (tuned) RMSE: **99.81** vs. Linear Regression baseline of **119.06**
- 📊 Top 3 price drivers identified: **bedrooms, bathrooms, proximity to downtown**
- 📈 11 visualizations produced for executive-level communication

## Tech Stack
- **Language:** Python
- **Libraries:** Scikit-learn, Pandas, Matplotlib, Seaborn, NumPy
- **Models:** Random Forest, Linear Regression, GridSearchCV

## Features Engineered
- Distance to downtown (geospatial calculation)
- Amenities count
- Host tenure (years active)
- Room type encoding
- Neighborhood clustering

## Project Structure
```
airbnb-price-prediction/
│
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks
│   ├── EDA.ipynb           # Exploratory Data Analysis
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
├── visualizations/         # Output charts and plots
└── README.md
```

## How to Run
```bash
# Clone the repository
git clone https://github.com/Vidhee843/airbnb-price-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/model_training.ipynb
```

## Results & Visualizations
The project includes box plots, scatter plots, and feature importance charts communicating findings to a business audience. Key insight: **location and physical size** are the strongest predictors of Airbnb pricing across all 4 cities.

## Team
Built as part of MISM 6212 — Data Mining & Machine Learning for Business at Northeastern University (Spring 2025). Team of 4.
