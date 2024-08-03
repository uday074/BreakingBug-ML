# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv("/kaggle/input/heart-disease-data/heart_disease_uci.csv")

# Print the first 5 rows of the dataframe
print(df.head())

# Exploring the data type of each column
print(df.info())

# Checking the data shape
print(df.shape)

# Summarize the age column
print(df['age'].describe())

# Plot the histogram of age column
sns.histplot(df['age'], kde=True, color="#FF5733")
plt.show()

# Plot the mean, median, and mode of age column using sns
sns.histplot(df['age'], kde=True)
plt.axvline(df['age'].mean(), color='Red', label='Mean')
plt.axvline(df['age'].median(), color='Green', label='Median')
plt.axvline(df['age'].mode()[0], color='Blue', label='Mode')
plt.legend()
plt.show()

# Print the value of mean, median, and mode of age column
print('Mean:', df['age'].mean())
print('Median:', df['age'].median())
print('Mode:', df['age'].mode()[0])

# Plot the histogram of age column using plotly and coloring by sex
fig = px.histogram(data_frame=df, x='age', color='sex')
fig.show()

# Find the values of sex column
print(df['sex'].value_counts())

# Calculate the percentage of male and female value counts in the data
male_count = df['sex'].value_counts()[1]
female_count = df['sex'].value_counts()[0]
total_count = male_count + female_count

male_percentage = (male_count / total_count) * 100
female_percentage = (female_count / total_count) * 100

print(f'Male percentage in the data: {male_percentage:.2f}%')
print(f'Female percentage in the data: {female_percentage:.2f}%')

difference_percentage = ((male_count - female_count) / female_count) * 100
print(f'Males are {difference_percentage:.2f}% more than females in the data.')

# Find the value counts of age column grouped by sex column
print(df.groupby('sex')['age'].value_counts())

# Plot the countplot of dataset column
fig = px.bar(df, x='dataset', color='sex')
fig.show()

# Print the values of dataset column grouped by sex
print(df.groupby('sex')['dataset'].value_counts())

# Plot the mean, median, and mode of age column grouped by dataset column
print("Mean of the dataset: ", df.groupby('dataset')['age'].mean())
print("Median of the dataset: ", df.groupby('dataset')['age'].median())
print("Mode of the dataset: ", df.groupby('dataset')['age'].agg(lambda x: pd.Series.mode(x)[0]))

# Value count of cp column
print(df['cp'].value_counts())

# Count plot of cp column by sex column
sns.countplot(data=df, x='cp', hue='sex')
plt.show()

# Count plot of cp column by dataset column
sns.countplot(data=df, x='cp', hue='dataset')
plt.show()

# Plot of age column grouped by cp column
fig = px.histogram(data_frame=df, x='age', color='cp')
fig.show()

# Summarize the trestbps column
print(df['trestbps'].describe())

# Dealing with missing values in trestbps column
print(f"Percentage of missing values in trestbps column: {df['trestbps'].isnull().sum() / len(df) * 100:.2f}%")

# Impute the missing values of trestbps column using Iterative Imputer
imputer1 = IterativeImputer(max_iter=10, random_state=42)
df['trestbps'] = imputer1.fit_transform(df[['trestbps']])

# Check the missing values in trestbps column
print(f"Missing values in trestbps column: {df['trestbps'].isnull().sum()}")

# Check data types or category of columns
print(df.info())

# Check for missing values
print((df.isnull().sum() / len(df) * 100).sort_values(ascending=False))

# Define missing data columns
missing_data_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()

# Define categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(exclude='object').columns.tolist()

print(f'Categorical Columns: {cat_cols}')
print(f'Numerical Columns: {num_cols}')

# Define a function to impute categorical missing data
def impute_categorical_missing_data(df, col):
    df_null = df[df[col].isnull()]
    df_not_null = df[df[col].notnull()]
    X = df_not_null.drop(col, axis=1)
    y = df_not_null[col]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputer.fit(X)
    X_null = df_null.drop(col, axis=1)
    y_pred = imputer.transform(X_null)
    df.loc[df[col].isnull(), col] = label_encoder.inverse_transform(np.round(y_pred).astype(int))
    return df

# Define a function to impute numerical missing data
def impute_numerical_missing_data(df, col):
    df_null = df[df[col].isnull()]
    df_not_null = df[df[col].notnull()]
    X = df_not_null.drop(col, axis=1)
    y = df_not_null[col]
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputer.fit(X)
    X_null = df_null.drop(col, axis=1)
    y_pred = imputer.transform(X_null)
    df.loc[df[col].isnull(), col] = y_pred
    return df

# Impute missing values using our functions
for col in missing_data_cols:
    print("Missing Values", col, ":", str(round((df[col].isnull().sum() / len(df)) * 100, 2)) + "%")
    if col in cat_cols:
        df = impute_categorical_missing_data(df, col)
    elif col in num_cols:
        df = impute_numerical_missing_data(df, col)

print(df.isnull().sum().sort_values(ascending=False))

# Remove rows with outlier values in trestbps column
df = df[df['trestbps'] != 0]

# Split the data into features and target variable
X = df.drop('num', axis=1)
y = df['num']

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define models
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('XGBoost Classifier', XGBClassifier(random_state=42)),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Naive Bayes Classifier', GaussianNB())
]

best_model = None
best_accuracy = 0.0

# Iterate over the models and evaluate their performance
for name, model in models:
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    categorical_cols = ['thal', 'ca', 'slope', 'exang', 'restecg', 'fbs', 'cp', 'sex']

def evaluate_classification_models(X, y, categorical_columns):
    # Encode categorical columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    # Train and evaluate models
    results = {}
    best_model = None
    best_accuracy = 0.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = name

    return results, best_model

# Example usage:
X = df.drop(columns=['num'])  # Assuming 'num' is the target column
y = df['num']

results, best_model = evaluate_classification_models(X, y, categorical_cols)
print("Model accuracies:", results)
print("Best model:", best_model)


def hyperparameter_tuning(X, y, categorical_columns, models):
    # Encode categorical columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Define dictionary to store results
    results = {}

    # Perform hyperparameter tuning for each model
    for model_name, model in models.items():
        # Define parameter grid for hyperparameter tuning
        param_grid = {}
        if model_name == 'Logistic Regression':
            param_grid = {'C': [0.1, 1, 10, 100]}
        elif model_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7, 9]}
        elif model_name == 'NB':
            param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
        elif model_name == 'SVM':
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
        elif model_name == 'Decision Tree':
            param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'Random Forest':
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'XGBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'GradientBoosting':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif model_name == 'AdaBoost':
            param_grid = {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200]}
        
        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Get best hyperparameters and evaluate on test set
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Store results in dictionary
        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

# Define models dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Example usage:
results = hyperparameter_tuning(X, y, categorical_cols, models)
for model_name, result in results.items():
    print("Model:", model_name)
    print("Best hyperparameters:", result['best_params'])
    print("Accuracy:", result['accuracy'])
    print()