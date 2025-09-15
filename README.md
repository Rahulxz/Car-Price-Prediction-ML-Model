# Car-Price-Prediction-ML-Model

**Project Overview**

This project focuses on building a machine learning model to predict car prices based on a dataset containing various car attributes. The goal is to develop a model that can accurately estimate the price of a car given its characteristics.

**Dataset**
The project utilizes the Car_Price_Prediction.csv dataset. This dataset includes the following columns:

**Dataset link**:https://drive.google.com/file/d/1nqxHNIXT3eFxWYqcMr-sWq6s1bo-EuxE/view?usp=sharing

**Make:** The brand of the car.

**Model:** The specific model of the car.

**Year:** The manufacturing year of the car.

**Engine Size:** The size of the car's engine.

**Mileage:** The distance the car has traveled.

**Fuel Type:** The type of fuel the car uses (e.g., Petrol, Diesel, Electric).

**Transmission:** The type of transmission (e.g., Manual, Automatic).

**Price:** The target variable, representing the price of the car.

The dataset contains 1000 entries and 8 columns, with no missing values as confirmed by the df.info() output.

**Data Exploration and Preprocessing**

Before building the model, the following data exploration and preprocessing steps were performed:

**Loading Data:** The dataset was loaded into a pandas DataFrame.

**Initial Inspection:** The first few rows (df.head()), shape (df.shape()), and data types and non-null counts (df.info()) were inspected to understand the structure and completeness of the data.

**Categorical Feature Encoding:** Categorical columns (Make, Model, Fuel Type, and Transmission) were encoded into numerical representations to make them suitable for machine learning algorithms. A simple manual encoding based on counts or alphabetical order was applied.

**Mileage Binning:** The Mileage feature was binned into discrete categories to potentially capture non-linear relationships and reduce the impact of outliers.

**Numerical Feature Scaling:** Numerical features (Year, Engine Size, Mileage, and Mileage_Bin) were scaled using StandardScaler to standardize their ranges, which can improve the performance of many machine learning models.

**Model Training**
A Random Forest Regressor model was chosen for the car price prediction task.

**Data Splitting:** The dataset was split into training and testing sets (X_train, X_test, y_train, y_test) to evaluate the model's performance on unseen data.

**Model Initialization and Training:** A RandomForestRegressor model was initialized and trained on the training data (X_train, y_train).
Model Evaluation

The performance of the trained Random Forest Regressor model was evaluated using common regression metrics:

**Prediction:** Predictions (y_pred) were made on the test set (X_test).

**Metrics Calculation:** **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, and **R-squared (R²)** were calculated to quantify the model's accuracy and goodness of fit.

**Visualization:** A scatter plot of Actual Prices vs. Predicted Prices was generated to visually assess the model's performance and identify potential patterns in the residuals.

**Hyperparameter Tuning**

Hyperparameter tuning was performed using GridSearchCV to find the best combination of hyperparameters for the Random Forest Regressor model within a defined search space.

**Parameter Grid Definition:** A dictionary (param_grid) was created specifying the hyperparameters (n_estimators, max_depth) and their respective values to be explored. 

**Grid Search Implementation:** GridSearchCV was used to systematically search for the best hyperparameters based on a specified scoring metric (negative mean squared error) and cross-validation.
 
**Best Parameters and Score:** The best hyperparameters found and the corresponding cross-validation score were identified.

**Tuned Model Evaluation:** A new Random Forest model was trained using the best hyperparameters and evaluated on the test set to compare its performance against the initial model.

**Results and Insights**

The evaluation metrics and the scatter plot indicate that the Random Forest Regressor model is capable of predicting car prices with a reasonable degree of accuracy (R-squared around 0.80). Hyperparameter tuning within the explored range resulted in minor changes in the evaluation metrics, suggesting that the initial default parameters were already performing quite well or that a wider search space or different hyperparameters need to be considered for further improvement.
