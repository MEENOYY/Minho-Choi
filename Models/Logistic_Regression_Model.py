#Decision Logistic Regression Model
#Author: Minho Choi

# !pip install ucimlrepo
# !pip install joblib

from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from IPython.display import display


# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

df_X= pd.DataFrame(X , columns = heart_disease.feature_names)
df_y = pd.DataFrame(y , columns = heart_disease.target_names)
data = pd.concat([df_X , df_y] , axis=1)

data.to_csv('output.csv', index=False)

# Feature selection and target
selected_features = ['thalach', 'oldpeak', 'ca', 'chol', 'thal']
target = 'num'

# Handle missing values
data.dropna(subset=selected_features + [target], inplace=True)

# Binary classification for the target (0: No heart disease, 1: heart disease)
data[target] = data[target].apply(lambda x: 1 if x > 0 else 0)

# Split data into features (X) and target (y)
X = data[selected_features]
y = data[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning: Define valid parameter grid (include only valid combinations of penalty, solver, and l1_ratio)
param_grid = [
    {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced']},
    {'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs', 'saga'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced']},
    {'penalty': ['elasticnet'], 'solver': ['saga'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'class_weight': [None, 'balanced'], 'l1_ratio': [0.5]},
]

# Logistic Regression model
model = LogisticRegression(random_state=42)

# Perform grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=1,
    n_jobs=-1,  # Use all available cores
    error_score='raise' # help debug invalid parameter combinations during fits
)

# Fit grid search to the training data
grid_search.fit(X_train, y_train)

# Display the best parameters and cross-validation score
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

# Use the best model
best_model = grid_search.best_estimator_


# Make predictions with the best model (default threshold: 0.5)
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Predict probabilities for the positive class
y_prob = best_model.predict_proba(X_test)[:, 1]

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute the AUC score
roc_auc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.show()

# Adjust threshold based on ROC curve
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
y_pred_adjusted = (y_prob >= optimal_threshold).astype(int)

# Evaluate the model with adjusted threshold
print(f"Optimal Threshold: {optimal_threshold:.2f}")
print("Confusion Matrix (Adjusted Threshold):")
print(confusion_matrix(y_test, y_pred_adjusted))

print("\nClassification Report (Adjusted Threshold):")
print(classification_report(y_test, y_pred_adjusted))

print("\nAccuracy Score (Adjusted Threshold):")
print(accuracy_score(y_test, y_pred_adjusted))


from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

df_X= pd.DataFrame(X , columns = heart_disease.feature_names)
df_y = pd.DataFrame(y , columns = heart_disease.target_names)
data = pd.concat([df_X , df_y] , axis=1)
display(data)