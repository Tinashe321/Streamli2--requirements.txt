#!/usr/bin/env python
# coding: utf-8

# # Question1
# 

# In[1]:


import pandas as pd
import numpy as np
import sqlite3 as sql
import csv
import sqlite3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# # Question 3

# In[ ]:





# In[2]:


connection = sql.connect("HeartDatabase.db")


# In[3]:


cursor = connection.cursor()


# In[4]:


import sqlite3
import pandas as pd

# Create a connection to the database (will create if it doesn't exist)
conn = sqlite3.connect('Heart.db') 

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# Check if the table already exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Health'")
table_exists = cursor.fetchall()

# If the table does not exist, create it
if not table_exists:
    cursor.execute('''
        CREATE TABLE Health(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            sex INTEGER,
            cp  INTEGER,
            trestbps INTEGER,
            chol  INTEGER,
            fbs   INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak FLOAT,
            slope INTEGER,
            ca INTEGER,
            that INTEGER,
            target INTEGER
        )
    ''')

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("C:\\Users\\USER\\Downloads\\HEARTS.csv", header=0, sep=';')

# Load the data from the DataFrame into the SQLite database
df.to_sql('Health', conn, if_exists='replace', index=False)


# In[5]:


# Read the CSV file into a pandas DataFrame

df = pd.read_csv("C:\\Users\\USER\\Downloads\\HEARTS.csv", header=0, sep=';')
df


# In[6]:


cursor.execute("SELECT * FROM Health")
rows = cursor.fetchall()
for row in rows:
    print(row)


# In[7]:


# 2.1.a: Data Cleaning & Preprocessing

# Example Preprocessing Steps (Add more as needed):
# 1. Check for duplicate rows:
print("Duplicates: ", df.duplicated().sum())
# 2. Drop duplicate rows
df.drop_duplicates(inplace=True)

# Handle missing values 
print("Missing values: ", df.isnull().sum())
for column in df.columns:
    # Replace missing values with the mean of that column if it is numeric
    if df[column].dtype in ['int64', 'float64']:
        df[column].fillna(df[column].mean(), inplace=True)
    # Replace missing values with the mode of that column if it is not numeric
    else:
        df[column].fillna(df[column].mode()[0], inplace=True)


# In[8]:


# Import the necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Sample DataFrame (replace this with your actual DataFrame)
data = {
    'sex': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'cp': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
    'fbs': ['High', 'Low', 'High', 'High', 'Low', 'Low', 'High', 'High'],
    'restecg': ['Normal', 'Abnormal', 'Normal', 'Abnormal', 'Normal', 'Abnormal', 'Normal', 'Abnormal'],
    'exang': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes'],
    'slope': ['Up', 'Down', 'Up', 'Down', 'Up', 'Down', 'Up', 'Down'],
    'ca': ['0', '1', '0', '2', '1', '0', '3', '2'],
    'thal': ['Normal', 'Abnormal', 'Normal', 'Abnormal', 'Normal', 'Abnormal', 'Normal', 'Abnormal'],
    'target': [1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# Convert categorical variables to the 'category' data type
categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for var in categorical_variables:
    df[var] = df[var].astype('category')

# Convert the 'target' variable to the 'category' data type
df['target'] = df['target'].astype('category')

# Create a subplot grid 
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))

# Plot distributions for each categorical variable
for i, var in enumerate(categorical_variables):
    row = i // 2
    col = i % 2
    sns.countplot(x=var, hue='target', data=df, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {var} by Target')

plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure 'df' DataFrame is loaded correctly (from previous steps)

# Convert categorical variables to the 'category' data type (optional, but good practice)
categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for var in categorical_variables:
    if df[var].dtype != 'category':  
        df[var] = df[var].astype('category') 

# Create a figure with 8 subplots (4 rows, 2 columns)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))
fig.suptitle('Distribution of Categorical Variables by Target', fontsize=16)

# Plot the distributions for each categorical variable
for i, var in enumerate(categorical_variables):
    row = i // 2  # Calculate row index
    col = i % 2   # Calculate column index
    sns.countplot(x=var, hue='target', data=df, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {var} by Target', fontsize=12)
    axes[row, col].set_xlabel(var, fontsize=10)  # Optional: Adjust x-axis label
    axes[row, col].set_ylabel('Count', fontsize=10) # Optional: Adjust y-axis label

# Adjust layout for better spacing
plt.tight_layout(pad=2.0)  # Increase spacing between subplots 
plt.show()

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

# Ensure 'df' DataFrame is loaded correctly (from previous steps)

# Convert categorical variables to the 'category' data type (optional, but good practice)
categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for var in categorical_variables:
    if str(df[var].dtype) != 'category':  # Convert var to string type for comparison
        df[var] = df[var].astype('category') 

# Create a figure with 8 subplots (4 rows, 2 columns)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))
fig.suptitle('Distribution of Categorical Variables by Target', fontsize=16)

# Plot the distributions for each categorical variable
for i, var in enumerate(categorical_variables):
    row = i // 2  # Calculate row index
    col = i % 2   # Calculate column index
    sns.countplot(x=var, hue='target', data=df, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {var} by Target', fontsize=12)
    axes[row, col].set_xlabel(var, fontsize=10)  # Optional: Adjust x-axis label
    axes[row, col].set_ylabel('Count', fontsize=10) # Optional: Adjust y-axis label



print(df)


# Adjust layout for better spacing
plt.tight_layout(pad=2.0)  # Increase spacing between subplots 
plt.show()


# In[10]:


# 2.1.c: Distribution of Numeric Variables by Target 

# Identify numeric columns that are not the target variable
numeric_variables = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_variables.remove('target')

# Create a subplot grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))  # Adjust figsize as needed

# Plot distributions for each numeric variable
for i, var in enumerate(numeric_variables):
    row = i // 3
    col = i % 3
    sns.histplot(data=df, x=var, hue='target', kde=True, element="step", ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {var} by Target')

plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ... (Your code to create the 'df' DataFrame) ...

# 1. More Flexible Data Type Selection
numeric_variables = df.select_dtypes(include='number').columns.tolist() # includes all numeric types
# 2. Conditional Removal of 'target'
if 'target' in numeric_variables:
    numeric_variables.remove('target')

# 3. Dynamic Subplot Grid
num_plots = len(numeric_variables)
num_rows = (num_plots + 2) // 3 # Ensure at least 2 rows
num_cols = 3

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))

# 4. Handle Extra Subplots (if any)
for i in range(num_plots, num_rows * num_cols):
    fig.delaxes(axes.flatten()[i]) # Remove unused subplots

# 5. Plot Distributions
for i, var in enumerate(numeric_variables):
    row = i // 3
    col = i % 3
    sns.histplot(data=df, x=var, hue='target', kde=True, element="step", ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {var} by Target')

plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd

# Assuming heart_data_cleaned is your preprocessed DataFrame
# Replace the placeholders with your actual preprocessed data
data = {
    'age': [63, 37, 41, 56, 57],
    'sex': [1, 1, 0, 1, 0],
    # Add more columns as per your actual preprocessed data
    'target': [1, 0, 1, 1, 0]  # Assuming 'target' is the target variable
}

heart_data_cleaned = pd.DataFrame(data)

# Now you can proceed with the rest of the code using the 'heart_data_cleaned' DataFrame


# In[ ]:


import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Assuming heart_data_cleaned is your preprocessed DataFrame
X = heart_data_cleaned.drop('target', axis=1)
y = heart_data_cleaned['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numeric and categorical features
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
numeric_transformer = StandardScaler()

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create and train the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

model.fit(X_train, y_train)

# Save the model
model_filename = r"C:\Users\USERS\Downloads\HEARTS.csv"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {model_filename}")


# # 3.1  Get your data ready for fitting a machine learning model on it by performing the appropriate preprocessing techniques.
# 
#  

# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the data
file_path = r'C:\Users\USER\Downloads\HEARTS.csv'
data = pd.read_csv(file_path, delimiter=';')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Identify features and target
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical features (scaling and imputing missing values)
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features (encoding and imputing missing values)
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Display the shape of the preprocessed data
print("Shape of preprocessed training data:", X_train_preprocessed.shape)
print("Shape of preprocessed testing data:", X_test_preprocessed.shape)


# # 3.2. Select 3 appropriate machine learning models for your heart disease prediction problem. Provide a short explanation of each chosen model as well as two advantages and disadvantages of each. Use the three models to fit your data and perform predictions on it, then determine which model performs the best. Save the model to disk. Remember, this saved model will then be used to model your decision support system.
# 
# # (20 marks)

# Selected Machine Learning Models for Heart Disease Prediction
# Logistic Regression
# 
# Explanation: Logistic Regression is a linear model used for binary classification tasks. It models the probability of the default class (e.g., the presence of heart disease) by applying a logistic function to a linear combination of the input features.
# Advantages:
# Simplicity: Easy to implement and understand, making it a good baseline model.
# Interpretability: The coefficients can provide insights into the importance of each feature.
# Disadvantages:
# Linearity: Assumes a linear relationship between the features and the log-odds of the outcome, which may not always be the case.
# Outliers: Sensitive to outliers which can affect the performance.
# Random Forest
# 
# Explanation: Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks. It combines the results of these trees to improve predictive accuracy and control overfitting.
# Advantages:
# Accuracy: Often more accurate than individual decision trees due to ensemble learning.
# Robustness: Less prone to overfitting and can handle large datasets with higher dimensionality.
# Disadvantages:
# Complexity: More complex to interpret compared to single decision trees.
# Resource Intensive: Requires more computational resources and memory.
# Support Vector Machine (SVM)
# 
# Explanation: SVM is a powerful classification algorithm that finds the hyperplane that best separates the classes in the feature space. It is effective in high-dimensional spaces and works well for both linear and non-linear classification tasks.
# Advantages:
# Effectiveness in High-Dimensions: Works well in high-dimensional spaces and is effective when the number of dimensions exceeds the number of samples.
# Robustness to Overfitting: Particularly effective with clear margin of separation.
# Disadvantages:
# Computation: Computationally intensive and may not scale well with very large datasets.
# Parameter Tuning: Requires careful tuning of parameters like the kernel and regularization parameters.

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the data
file_path = r'C:\Users\USER\Downloads\HEARTS.csv'
data = pd.read_csv(file_path, delimiter=';')

# Identify features and target
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical features (scaling and imputing missing values)
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features (encoding and imputing missing values)
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipelines for each model
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Fit and evaluate each model
pipelines = [log_reg_pipeline, rf_pipeline, svm_pipeline]
model_names = ['Logistic Regression', 'Random Forest', 'SVM']
best_model = None
best_accuracy = 0

for model, name in zip(pipelines, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.4f}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model to disk
joblib.dump(best_model, 'best_model.pkl')
print(f'Best model saved: {best_model.steps[-1][0]} with accuracy {best_accuracy:.4f}')


# # Training model using SVM Accuracy: 0.8689

# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the data
file_path = r'C:\Users\USER\Downloads\HEARTS.csv'
data = pd.read_csv(file_path, delimiter=';')

# Identify features and target
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical features (scaling and imputing missing values)
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features (encoding and imputing missing values)
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline that includes preprocessing and the SVM model
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, random_state=42))
])

# Train the SVM model
svm_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy:.4f}')

# Save the model to disk
model_save_path = r'C:\Users\USER\anaconda3\Lib\site-packages\streamlit\best_model.pkl'
joblib.dump(svm_pipeline, model_save_path)
print(f'Model saved to {model_save_path}')


# In[14]:


import streamlit as st
import joblib
import numpy as np 
import pandas as pd

# Load the trained model
model = joblib.load('C:\\Users\\USER\\anaconda3\\Lib\\site-packages\\streamlit\\best_model.pkl')

# Define the app
st.title('Heart Disease Prediction')
st.write('Enter the patient details to predict the likelihood of heart disease.')

# Input fields for patient data
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.number_input('Chest Pain Type', min_value=0, max_value=3, value=0)
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.number_input('Resting Electrocardiographic Results (restecg)', min_value=0, max_value=2, value=1)
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.number_input('Slope of the Peak Exercise ST Segment (slope)', min_value=0, max_value=2, value=1)
ca = st.number_input('Number of Major Vessels Colored by Flouroscopy (ca)', min_value=0, max_value=4, value=0)
thal = st.number_input('Thalassemia (thal)', min_value=0, max_value=3, value=2)

# Convert inputs into a dataframe
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs],
    'restecg': [restecg], 'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
    'slope': [slope], 'ca': [ca], 'thal': [thal]
})

# Predict and display result
if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of positive class

    if prediction == 1:
        st.error(f'The patient is likely to have heart disease. Probability: {prediction_proba:.2f}')
    else:
        st.success(f'The patient is not likely to have heart disease. Probability: {prediction_proba:.2f}')

# Add some information about the app
st.write("""
### About the App
This application uses a machine learning model to predict the likelihood of heart disease based on patient data. Please fill in all the fields with the patient's information to get a prediction.
""")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




