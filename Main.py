# Import the function from package_checker.py
from package_checker import check_and_install

# Run the check and install process
check_and_install()

# Importing necessary libraries
import pandas as pd    # For data manipulation and analysis
import numpy as np      # For numerical computations and array handling
import seaborn as sns   # For data visualization, built on top of Matplotlib
import matplotlib.pyplot as plt   # For creating static, animated, and interactive visualizations
from scipy import stats  # For statistical computations and tests

# Scikit-learn: Core libraries for machine learning models and utilities
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding labels and scaling data
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, GridSearchCV, cross_val_score, KFold  # For splitting data, model validation, and hyperparameter tuning

# Keras: High-level neural networks API, running on top of TensorFlow
from keras.models import Sequential, load_model  # For building and loading neural network models
from keras.layers import Dense  # For defining fully connected neural network layers

# Metrics and evaluation tools for model performance
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, classification_report, roc_curve, auc, precision_score, recall_score, f1_score

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression  # For logistic regression models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble methods for classification (Random Forest, Gradient Boosting)
from urllib.request import urlopen  # For fetching data from the web
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis  # For k-nearest neighbors and feature extraction
from sklearn.naive_bayes import GaussianNB  # For Naive Bayes classification
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree classification
from sklearn import tree, datasets, svm, metrics  # Additional Scikit-learn utilities for tree-based models, datasets, SVM, and metrics

# Graphviz: Visualization of decision trees and other graph structures
import graphviz

# TensorFlow: Deep learning framework, required by Keras
import tensorflow as tf
from tensorflow import keras  # Keras is part of TensorFlow since version 2.x

# Statistical tools from SciPy
from scipy.stats import ttest_ind  # For performing t-tests

# Dimensionality reduction using Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Plotly: Interactive graphing library
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff  # For creating data visualization plots

# Colormap utilities for visualizations
from matplotlib.colors import ListedColormap

# Set visualization style and display options
plt.style.use('ggplot')
pd.set_option('display.max_columns', 500)

# Import functions from other modules
from file_loader import load_data
from data_processing import process_data
from variable_relationships import analyze_relationships

# Path to the CSV file
csv_file_path = 'data.csv'

# Load the dataset using the function from file_loader.py
data = load_data(file_path=csv_file_path)

# Display the first few rows of the DataFrame to verify it was loaded correctly
print(data.head())

# Process the dataset using the function from data_processing.py
processed_data = process_data(file_path=csv_file_path)

# Example: Print the shape of the processed data
print(f"\nShape of the processed dataset: {processed_data.shape}")

# Analyze relationships and clean the dataset using the function from variable_relationships.py
data_final = analyze_relationships(processed_data)

# Display the cleaned dataset columns
print("\nRemaining columns after removing multicollinear variables:")
print(data_final.columns)

# Further analysis or model training can be added here
# Import the function from package_checker.py
from package_checker import check_and_install

# Run the check and install process
check_and_install()

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.offline as pyo
import plotly.graph_objs as go
from file_loader import load_data
from data_processing import process_data
from data_analysis import perform_data_analysis  # Assuming you created this function

# Load the dataset
data = load_data()

# Display the first few rows of the DataFrame to verify it was loaded correctly
print(data.head())

# Process the dataset
csv_file_path = 'data.csv'
processed_data = process_data(file_path=csv_file_path)

# Print the shape of the processed data
print(f"\nShape of the processed dataset: {processed_data.shape}")

# Perform data analysis and visualization
perform_data_analysis(processed_data)

from machine_learning import (preprocess_data, train_knn, train_svm, train_logreg,
                               train_decision_tree, train_random_forest,
                               train_gradient_boosting, train_naive_bayes,
                               train_neural_network, evaluate_model)

# Path to the CSV file
csv_file_path = 'data.csv'

# Load the dataset using the function from file_loader.py
data = load_data(file_path=csv_file_path)

# Display the first few rows of the DataFrame to verify it was loaded correctly
print(data.head())

# Process the dataset using the function from data_processing.py
processed_data = process_data(file_path=csv_file_path)

# Analyze relationships and clean the dataset using the function from variable_relationships.py
data_final = analyze_relationships(processed_data)

# Display the cleaned dataset columns
print("\nRemaining columns after removing multicollinear variables:")
print(data_final.columns)

# Preprocess the data for machine learning
X_train, X_test, y_train, y_test = preprocess_data(data_final)

# Train different models
knn_model = train_knn(X_train, y_train)
svm_model = train_svm(X_train, y_train)
logreg_model = train_logreg(X_train, y_train)
dt_model = train_decision_tree(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)
gb_model = train_gradient_boosting(X_train, y_train)
nb_model = train_naive_bayes(X_train, y_train)
nn_model = train_neural_network(X_train, y_train)

# Evaluate the models
print("KNN Model Evaluation:")
evaluate_model(knn_model, X_test, y_test)

print("SVM Model Evaluation:")
evaluate_model(svm_model, X_test, y_test)

print("Logistic Regression Model Evaluation:")
evaluate_model(logreg_model, X_test, y_test)

print("Decision Tree Model Evaluation:")
evaluate_model(dt_model, X_test, y_test)

print("Random Forest Model Evaluation:")
evaluate_model(rf_model, X_test, y_test)

print("Gradient Boosting Model Evaluation:")
evaluate_model(gb_model, X_test, y_test)

print("Naive Bayes Model Evaluation:")
evaluate_model(nb_model, X_test, y_test)

print("Neural Network Model Evaluation:")
evaluate_model(nn_model, X_test, y_test)