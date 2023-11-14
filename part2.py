import numpy as np
import matplotlib.pyplot as plt
#plt.ion()
import pandas as pd
from keras import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
import math
import shap
from os.path import join


# Write experiment settings in a file
def write_experiment_settings(experiment, experiment_settings):
    with open(experiment + '.txt', 'w') as f:
        for (name, val) in zip(experiment_settings.keys(), experiment_settings.values()):
            f.write(str(name) + ' = ' + str(val) + '\n')
    return


# Apply windowing on the time series data
def apply_windows(input_values, window_size=20):
    input_values_, target_values_ = [], []
    for i in range(len(input_values) - window_size):
        d = i + window_size
        input_values_.append(input_values[i:d])
        target_values_.append(input_values[d])
    return np.array(input_values_), np.array(target_values_)


# Split the dataset into train and test subsets
def split_train_test(input_values, target_values, test_percentage=0.2):
    n = math.ceil((1 - test_percentage) * len(input_values))

    input_values_train = input_values[0:n]
    target_values_train = target_values[0:n]

    input_values_test = input_values[n:]
    target_values_test = target_values[n:]

    return input_values_train, target_values_train, input_values_test, target_values_test


# Build model based on the specifications
def build_model(input_shape, activation, layers_number, neurons_number, loss, optimizer_name):
    if layers_number < 1:
        raise ValueError('The number of hidden layers should be at least 1.\n')

    if loss not in ['mean_absolute_error', 'mean_squared_error']:
        raise ValueError('Loss function should be {mean_squared_error, mean_absolute_error}.\n')

    if optimizer_name not in ['Adam', 'SGD']:
        raise ValueError('Optimizer should be {SGD, Adam}.\n')

    if len(neurons_number) != layers_number + 1:
        raise ValueError('You should provide the number of neurons for each layer. For '
                         + str(layers_number) + 'layers, please provide '
                         + str(layers_number) + 'values. \n')

    if activation not in ['relu', 'sigmoid']:
        raise ValueError('Activation function should be {relu, sigmoid}.\n')

    model = Sequential()
    for i in range(layers_number):
        if i == 0:
            model.add(Dense(neurons_number[i], input_dim=input_shape, activation=activation))
        else:
            model.add(Dense(neurons_number[i], activation=activation))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer_name, metrics=['mse', 'mae'], run_eagerly=True)
    return model


# Train and validate the model according to specifications
def train_and_validate_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    verbose = 0
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=(x_test, y_test))
    return model, history


# Plots the loss function of the model across epochs
def plot_model_loss(history, experiment):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Evolution of the Loss Function')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    save_path = join('report', 'images', experiment + '_model_loss.png')
    plt.savefig(save_path)
    #plt.show()


# Plots the predicted result
def plot_prediction(y_test, predict_test, experiment):
    len_prediction = [x for x in range(len(y_test))]
    plt.figure()
    plt.plot(len_prediction, y_test, marker='.', label="actual")
    plt.plot(len_prediction, predict_test, 'r', label="prediction")
    plt.subplots_adjust(left=0.07)
    plt.title('Comparison of Prediction Results')
    plt.ylabel('Values', size=15)
    plt.xlabel('Prediction Instances', size=15)
    plt.legend(fontsize=15)
    save_path = join('report', 'images', experiment + '_predictions.png')
    plt.savefig(save_path)
    #plt.show()


# Plot SHAP values
def plot_shap_values(model, x_test, experiment):
    plt.figure()
    explainer = shap.Explainer(model.predict, x_test)
    shap_values = explainer(x_test)
    shap.plots.bar(shap_values, show=False)
    save_path = join('report', 'images', experiment + '_shap_values.png')
    plt.savefig(save_path)
    #plt.show()


# Scale back predicted values
def scale_back_values(values, scaler):
    df_values = pd.DataFrame(values, columns=['value'])
    df_rescaled = scaler.inverse_transform(df_values)
    return df_rescaled


# Builds and evaluates the model
def build_and_evaluate_model(experiment, x_train, y_train, x_test, y_test, scaler, input_shape, activation,
                             layers_number, neurons_number, loss, optimizer_name, epochs, batch_size):
    model = build_model(input_shape, activation, layers_number, neurons_number, loss, optimizer_name)
    model, history = train_and_validate_model(model, x_train, y_train, x_test, y_test, epochs, batch_size)

    train_score = model.evaluate(x_train, y_train, verbose=0)

    print('\n')
    print('During training: Root Mean Squared Error(RMSE): %.5f, Mean Absolute Error(MAE) : %.5f ' % (
        np.sqrt(train_score[1]), train_score[2]))
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('During testing: Root Mean Squared Error(RMSE): %.5f, Mean Absolute Error(MAE) : %.5f ' % (
        np.sqrt(test_score[1]), test_score[2]))
    print('\n')

    plot_model_loss(history, experiment)

    predict_test = model.predict(x_test)
    y_test = scale_back_values(y_test, scaler)
    predict_test = scale_back_values(predict_test, scaler)
    plot_prediction(y_test, predict_test, experiment)

    plot_shap_values(model, x_test, experiment)

    return model
