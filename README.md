# Solar-Panel-Efficiency-Prediction

🛠️ Tools and Libraries Used
- Python (3.11)
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for preprocessing and evaluation
- XGBoost for building the regression model
- Google Colab as the development environment

📊 Approach
1. Data Loading:
   All files (train.csv, test.csv, and sample_submission.csv) were loaded using pandas.read_csv().

2. Initial Cleaning:
   - Dropped the id column from the training set as it’s not useful for modeling.
   - Converted non-numeric values (e.g., 'error') to NaN using pd.to_numeric(errors='coerce').
   - Filled missing values with 0 since this is a sensor dataset and missing data likely implies failure or drop-out.

3. Feature Engineering:
   - Created a new feature power as voltage × current, inspired by basic electrical principles.
   - Encoded categorical features like string_id, installation_type, and error_code using LabelEncoder.

4. Feature Scaling:
   Standardized the features using StandardScaler to ensure the model is not biased toward features with larger magnitudes.

5. Model Selection:
   Chose XGBoost for its robustness, ability to handle tabular data, and built-in regularization.

6. Training Strategy:
   - Split the dataset into training and validation sets using an 80-20 ratio.
   - Used XGBoost's early stopping to avoid overfitting.
   - Optimized on RMSE and evaluated using a custom score function scaled to a 0-100 range.

7. Prediction:
   - The final model was used to generate predictions on the test.csv file.
   - The output was stored in the format expected in sample_submission.csv.

📁 Files Included
- train.csv – Historical sensor data with efficiency values
- test.csv – Data without efficiency (to be predicted)
- sample_submission.csv – Submission format
- predict1_submission.csv – Final predictions
