import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.under_sampling import RandomUnderSampler

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

## Data Preprocessing

# Load data
data = pd.read_excel("CCD.xls", header=1)
data.columns = data.columns.astype(str)

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply undersampling to the training set
undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Create an undersampled validation set
X_val_resampled, y_val_resampled = undersampler.fit_resample(X_test, y_test)

## Building model

def buildModel(X_test, y_test):
    model = Sequential()

    # Hidden layers
    model.add(Dense(64, input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Adjust the learning rate
    optimizer = Adam(learning_rate=0.001)

    # Compiling model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(X_train_resampled, y_train_resampled, epochs=64, batch_size=30, validation_data=(X_test, y_test), callbacks=[early_stopping])

    return model, history

def drawModelGraph(history, validationName):
    plt.figure(figsize=(12, 4))


    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', validationName])


    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', validationName])


    plt.show()

def evaluateModel(model, history, validationName):
    # Print accuracy
    accuracy = model.evaluate(X_test, y_test)[1]
    print(f"Test Accuracy: {accuracy}")

    # Classification report
    y_pred = model.predict([X_test, y_test]).round()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Create confusion matrix using percentages
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Compute percentages for each cell in the confusion matrix
    conf_matrix_percent = conf_matrix / np.sum(conf_matrix, axis=1)[:, np.newaxis] * 100

    # Define labels for confusion matrix cells
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

    # Display the confusion matrix as a table with percentages and labels
    print("Confusion Matrix (Percentages):")
    print("               Predicted Non-default   Predicted Default")
    print("Actual Non-default        {:.2f}%              {:.2f}%".format(conf_matrix_percent[0][0], conf_matrix_percent[0][1]))
    print("Actual Default            {:.2f}%              {:.2f}%".format(conf_matrix_percent[1][0], conf_matrix_percent[1][1]))

    drawModelGraph(history, validationName)


model, history = buildModel(X_test, y_test)
model2, underSampledHistory = buildModel(X_val_resampled, y_val_resampled)
evaluateModel(model, history, 'Validation')
drawModelGraph(underSampledHistory, 'Validation (undersampled)')

