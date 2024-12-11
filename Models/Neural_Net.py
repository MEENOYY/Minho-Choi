# Artificial Neural Network Model
# Author: Sam Herring

from ucimlrepo import fetch_ucirepo
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor


def display_random_samples(df, n=5):
    for column in df.columns:
        print(f"Random samples from column '{column}':")
        print(df[column].sample(n=n).to_list())  # Adjust 'random_state' for consistent results
        print("-" * 40)

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

dict = {}
for col in X.columns:
   dict[col] = X[col].values.tolist()

for col in y.columns:
   dict[col] = y[col].values.tolist()

ca_counter = 0
ca_nan_inds = []
thal_counter = 0
thal_nan_inds = []
for i in range(len(dict['ca'])):
   if dict['ca'][i] != dict['ca'][i]:
       ca_counter += 1
       ca_nan_inds.append(i)
   if dict['thal'][i] != dict['thal'][i]:
       thal_counter += 1
       thal_nan_inds.append(i)

heart_data = pd.DataFrame(dict)

print(f'there are a total of {heart_data.shape[0]} rows in the dataset')
to_drop = ca_nan_inds + thal_nan_inds
print(f'dropping rows {to_drop}')
heart_data = heart_data.drop(to_drop, axis=0)
print(f'there are a total of {heart_data.shape[0]} rows in the dataset')

# eliminate age, since its effects get represented in the other variables
# eliminate sex for same reason
# eliminate trestbps because it seems to display low correlation to the target
# eliminate chol for same reason
# eliminate fbs for same reason
# eliminate restecg for same reason
# eliminate slope because low correlation to target and probably redundant with oldpeak
heart_data = heart_data.drop(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'slope'], axis=1)

print(f'The remaining columns are: {heart_data.columns.values}')
display_random_samples(heart_data)


mmscaler = MinMaxScaler(feature_range=(-1, 1))
y = heart_data['num'].values
#y = mmscaler.fit_transform(y)
X = heart_data.loc[:, heart_data.columns != 'num']
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def get_regularizer(reg_type, reg_value):
    if reg_type == 'l1':
        return tf.keras.regularizers.l1(reg_value)
    elif reg_type == 'l2':
        return tf.keras.regularizers.l2(reg_value)
    elif reg_type == 'l1_l2':
        return tf.keras.regularizers.l1_l2(l1=reg_value, l2=reg_value)
    else:
        return None

def create_model(reg_type, reg_value): #, momentum):
    regularizer = get_regularizer(reg_type, reg_value)
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(3,
                                    activation='gelu',
                                    kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(1,
                                    #activation='sigmoid',
                                    kernel_regularizer=regularizer))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def create_default_model(input_size):
    seed = int(time.time())
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    model.add(tf.keras.layers.Dense(3, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# to do further feature selection, remove one column and train ten times for each removal, then compare the losses
# this found that dropping 'cp' improved performance
results = []
for i in range(10):
    results_i = []
    for col in range(X_train.shape[1]):
        X_temp = np.delete(X_train, col, 1)
        X_test_temp = np.delete(X_test, col, 1)
        model_temp = create_default_model(X_temp.shape[1])
        model_temp.fit(X_temp, y_train,
                        validation_data=(X_test_temp, y_test),
                        epochs=200,
                        batch_size=80,
                        verbose=0)
        test_loss = model_temp.evaluate(X_test_temp, y_test, batch_size=80)
        results_i += (col, test_loss)
        print(f'col {col} done')
    results.append(results_i)
    print(f'iteration {i} done')
print(results)

model = KerasRegressor(model=create_model, verbose=0)

param_grid = {
    #'model__momentum': [0.01, 0.5, 0.99],
    'model__reg_type': ['l1', 'l2'],
    'model__reg_value': [0.0, 0.1, 0.3, 1.0],
    'batch_size': [40, 80, 120],
    'epochs': [200]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
grid_result = grid.fit(X_train, y_train)


# Print the best momentum and the corresponding score
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Score: {-grid_result.best_score_:.4f}")

best_model = create_model(grid_result.best_params_['model__reg_type'], grid_result.best_params_['model__reg_value'])

history = best_model.fit(X_train, y_train,
               validation_data=(X_test, y_test),
               epochs=200,
               batch_size=grid_result.best_params_['batch_size'])

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(history.history['loss'][-1])
print(best_model.layers[0].get_weights()[0])