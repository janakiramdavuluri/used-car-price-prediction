# Used Car Price Prediction

## Project Overview

This project predicts the selling price of used cars using machine learning techniques.

The project includes data preprocessing, exploratory data analysis, feature engineering, feature selection, and comparison of multiple regression models. The models are evaluated using appropriate regression metrics, and hyperparameter tuning is applied to improve model performance.

The goal is to build a reliable model that can estimate the price of a used car based on its available features.


## Dataset

The project uses the **CAR DETAILS FROM CAR DEKHO** dataset.

The dataset contains information about used cars and their selling prices. The features used in the project include:

* Car name
* Brand
* Fuel type
* Seller type
* Transmission
* Owner
* Kilometers driven
* Vehicle age

**Target variable:** Selling price of the used car.

The dataset is stored in:

`data/raw/CAR DETAILS FROM CAR DEKHO.csv`


## Machine Learning Workflow

The project follows these main steps:

1. **Data Loading** – Load the used-car dataset using Pandas.
2. **Data Understanding** – Inspect the dataset structure, data types, and statistical information.
3. **Exploratory Data Analysis (EDA)** – Analyze the data to understand feature distributions and relationships.
4. **Data Preprocessing** – Handle categorical and numerical features appropriately.
5. **Feature Engineering** – Create useful features such as vehicle age where applicable.
6. **Feature Selection** – Select relevant features for price prediction.
7. **Model Building** – Train multiple regression models.
8. **Model Evaluation** – Compare models using regression evaluation metrics.
9. **Hyperparameter Tuning** – Tune the selected model to improve its performance.
10. **Prediction** – Use the trained model to predict the price of an unseen used car.


## Models Used

The following regression models are used and compared in the project:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Extra Trees Regressor
* Tuned Random Forest Regressor

The models are compared based on their regression performance, and the best-performing model is selected for the final prediction.


## Evaluation Metrics

The models are evaluated using the following metrics:

* **MAE (Mean Absolute Error)** – Measures the average absolute difference between the actual and predicted prices.
* **RMSE (Root Mean Squared Error)** – Measures the square root of the average squared prediction error.
* **R² Score** – Measures how well the model explains the variation in the target variable.

For MAE and RMSE, **lower values are better**. For R², a value closer to **1 is better**.


## Project Structure

```text
used-car-price-prediction/
│
├── data/
│   └── raw/
│       └── CAR DETAILS FROM CAR DEKHO.csv
│
├── notebooks/
│   └── used_car_price_prediction.ipynb
│
├── src/
│
├── models/
│
├── outputs/
│   └── figures/
│
├── app/
│
├── README.md
├── requirements.txt
└── .gitignore
```

The notebook contains the complete exploratory analysis, preprocessing, model training, evaluation, tuning, and prediction workflow.


## Technologies Used

* **Python**
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Data visualization
* **Scikit-learn** – Machine learning, preprocessing, model training, evaluation, and hyperparameter tuning
* **Jupyter Notebook** – Development and experimentation


## Installation & Setup

### 1. Clone the repository

```text
git clone <your-github-repository-url>
cd used-car-price-prediction
```

### 2. Install the required libraries

```text
pip install -r requirements.txt
```

### 3. Run the notebook

Open the following notebook using Jupyter Notebook, JupyterLab, or Google Colab:

```text
notebooks/used_car_price_prediction.ipynb
```

Make sure the dataset is available at:

```text
data/raw/CAR DETAILS FROM CAR DEKHO.csv
```
