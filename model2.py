import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Input, Dense, LSTM, concatenate, Dropout, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Load the data with the correct header row
data = pd.read_excel("CCD.xls", header=1)
data.columns = data.columns.astype(str)

# Separate features (X) and target variable (y)
X_non_temporal = data.iloc[:, :5]  # Non-temporal features
X_temporal = data.iloc[:, 5:]  # Temporal features
y = data['default payment next month']

# Standardize the data
scaler = StandardScaler()
X_non_temporal = scaler.fit_transform(X_non_temporal)

# Reshape the temporal data for LSTM
X_temporal = X_temporal.values.reshape((X_temporal.shape[0], 1, X_temporal.shape[1]))

# Train-Test Split
X_train_nt, X_test_nt, X_train_temporal, X_test_temporal, y_train, y_test = train_test_split(
    X_non_temporal, X_temporal, y, test_size=0.2, random_state=42
)

# Build the model
input_nt = Input(shape=(X_train_nt.shape[1],))
dense_layer1 = Dense(10, activation='relu')(input_nt)
dense_layer1 = Dropout(0.5)(dense_layer1)

input_temporal = Input(shape=(X_train_temporal.shape[1], X_train_temporal.shape[2]))
lstm_layer1 = LSTM(128, activation='tanh', return_sequences=True)(input_temporal)
lstm_layer1 = Dropout(0.5)(lstm_layer1)
lstm_layer2 = LSTM(64, activation='tanh', return_sequences=True)(lstm_layer1)
lstm_layer2 = Dropout(0.4)(lstm_layer2)
lstm_layer3 = LSTM(32, activation='tanh', return_sequences=True)(lstm_layer2)
lstm_layer3 = Dropout(0.4)(lstm_layer3)


# Concatenate the outputs
merged = concatenate([Flatten()(dense_layer1), Flatten()(lstm_layer3)])

output = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[input_nt, input_temporal], outputs=output)

# Adjust the learning rate
optimizer = Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    [X_train_nt, X_train_temporal], y_train,
    epochs=50, batch_size=32,
    validation_data=([X_test_nt, X_test_temporal], y_test),
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_accuracy = model.evaluate([X_test_nt, X_test_temporal], y_test)[1]
print(f"Test Accuracy: {test_accuracy}")

# Generate predictions on the test set
y_pred = model.predict([X_test_nt, X_test_temporal]).round()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
total_instances = np.sum(conf_matrix)

# Print Confusion Matrix (in percentages) with labels
print("Confusion Matrix (in percentages):")
print("---------------------------")
print(f"|   {conf_matrix[0, 0] / total_instances * 100:.1f}% ({conf_matrix[0, 0]})   |   {conf_matrix[0, 1] / total_instances * 100:.1f}% ({conf_matrix[0, 1]})   |   True Negative (TN)   |")
print("---------------------------")
print(f"|   {conf_matrix[1, 0] / total_instances * 100:.1f}% ({conf_matrix[1, 0]})   |   {conf_matrix[1, 1] / total_instances * 100:.1f}% ({conf_matrix[1, 1]})   |   True Positive (TP)   |")
print("---------------------------")
print(f"|   False Negative (FN)   |   False Positive (FP)   |")
print("---------------------------")
print(f"|   {conf_matrix[0, 1] / total_instances * 100:.1f}% ({conf_matrix[0, 1]})   |   {conf_matrix[1, 0] / total_instances * 100:.1f}% ({conf_matrix[1, 0]})   |   FN   |")
print("---------------------------")
print(f"|   {conf_matrix[1, 1] / total_instances * 100:.1f}% ({conf_matrix[1, 1]})   |   {conf_matrix[0, 0] / total_instances * 100:.1f}% ({conf_matrix[0, 0]})   |   FP   |")
print("---------------------------")


# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Visualize training and validation metrics
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation', 'Test'])

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.show()