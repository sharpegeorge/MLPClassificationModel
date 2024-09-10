import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, concatenate
import numpy as np

# Load your dataset (assuming 'CCD.xls' as the file name with header in the first row)
data = pd.read_excel('CCD.xls', header=1)

# Ignore the last column
features = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Reshape the features for sequential input
num_categories = 10  # Number of unique categories in PAY (adjust based on your data)
embedding_dim = 5  # Dimension of the embedding space
seq_length = 6  # Number of time steps

# Normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

# Convert y_train to a numpy array
y_train = np.array(y_train)

# Compute class weights manually
class_weights = {0: 0.5, 1: 0.5}  # Adjust the weights based on your preference

# Flatten the time steps before oversampling
X_train_reshaped = X_train.reshape((len(X_train), -1))

# Use RandomOverSampler for oversampling the minority class
oversampler = RandomOverSampler(sampling_strategy='minority')
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_reshaped, y_train)

# Reshape the resampled data back to the original shape
X_train_resampled = X_train_resampled.reshape((len(X_train_resampled) // seq_length, seq_length, X_train_resampled.shape[1]))

# Build the model with dense layers before LSTM layers, dropout, and L2 regularization
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(seq_length, X_train_resampled.shape[2])))
model.add(Dropout(0.5))  # Add dropout layer
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))  # Add dropout layer
model.add(LSTM(units=128, return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))  # Add dropout layer
model.add(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.01)))  # Adjusted for return_sequences
model.add(Dropout(0.5))  # Add dropout layer
model.add(LSTM(units=32, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))  # Add dropout layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model with adjusted class weights
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping and adjusted class weights
history = model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=20, validation_split=0.2,
                    callbacks=[early_stopping], class_weight=class_weights)

# Evaluate the model on the test set
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot training and validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()