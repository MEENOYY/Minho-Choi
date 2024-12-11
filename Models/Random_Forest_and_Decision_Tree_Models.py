#Decision Tree and Random Forest Models
#Author: Miles Bardin

#!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
from numpy import array
import pandas as pd
import random
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Callable class of the Random Forest model for ease of comparing with other models
class RandomForest:
    def __init__(self, data, target, test_size=0.2, max_features='sqrt', num_estimators=10, rand_state=42):
        self.data = data
        self.target = target
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=rand_state)
        self.model = RandomForestClassifier(max_features=max_features, n_estimators=num_estimators, random_state=rand_state)
        self.model.fit(X_train, y_train)

    def predict(self, df):
        y_pred_rf = self.model.predict(df)
        return y_pred_rf

    def test(self, df, y_test):
        y_pred_rf = self.model.predict(df)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf)
        f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
        print(f'\nRandom Forest Precision: {precision_rf:.2f}')
        print(f'Random Forest Recall: {recall_rf:.2f}')
        print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
        print(f'Random Forest F1: {f1_rf:.2f}')
        return y_pred_rf

# OBTAIN AND CONFIGURE THE DATASET
# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X_all_attrs = heart_disease.data.features
y = heart_disease.data.targets

# Replace nonzero target values with 1
for i in range(len(y.values)):
    if y.values[i][0] > 0:
        y.values[i] = array([1])

# Check I updated the values correctly
# for val in y.values:
#     if type(val) != numpy.ndarray:
#         print(type(val))
#         raise TypeError

X_data = X_all_attrs.values
X_data = X_data[:, [4, 7, 9, 11, 12]]
X_cols = ['chol', 'thalac', 'oldpeak', 'ca', 'thal']

# Finding and removing outliers
outliers = []

for i in range(len(X_data)):
    if X_data[i][2] > 5:
        outliers.append(i)
    elif X_data[i][0] > 500:
        outliers.append(i)


X = pd.DataFrame(X_data, columns=X_cols)

X = X.drop(outliers)
y = y.drop(outliers)
y = numpy.ravel(y)

# Check I updated the values correctly
# for val in X.values:
#     if type(val) != numpy.ndarray:
#         print(type(val))
#         raise TypeError


# HYPER-PARAMETERS
n_features = X.shape[1]
size = 0.2                       # Tried with 0.25 and 0.3, but yielded worse results than 0.2
rand_state = random.randint(1, 10000)
feature = 'sqrt'                 # square root, the number of features divided by 3 then rounded up, and all features
                                     # Tried None, but yielded worse results than the others
                                     # No difference between 'sqrt' and 2, so arbitrarily choose 'sqrt'
estimators = 10                  # Tried 1, 5, 20, 25, 35, & 50 estimators, but didn't work as well as 10


# TEST FOR A SINGLE DECISION TREE
# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=rand_state)

# Train a single decision tree
tree = DecisionTreeClassifier(random_state=rand_state)
tree.fit(X_train, y_train)

# Predict and evaluate
y_pred_tree = tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
f1_tree = 2 * (precision_tree * recall_tree) / (precision_tree + recall_tree)
print(f'\n\nSingle Decision Tree Precision: {precision_tree:.2f}')
print(f'Single Decision Tree Recall: {recall_tree:.2f}')
print(f'Single Decision Tree Accuracy: {accuracy_tree:.2f}')
print(f'Single Decision Tree F1: {f1_tree:.2f}')

# TEST RANDOM FOREST MODEL
# Create a Random Forest model
random_forest = RandomForestClassifier(max_features=feature, n_estimators=estimators, random_state=rand_state)
random_forest.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)
print(f'\nRandom Forest Precision: {precision_rf:.2f}')
print(f'Random Forest Recall: {recall_rf:.2f}')
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
print(f'Random Forest F1: {f1_rf:.2f}')

# FEATURE IMPORTANCE
# Extract feature importances from the Random Forest model
feature_importances = random_forest.feature_importances_
features = X.columns

# Create a DataFrame for plotting
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()
