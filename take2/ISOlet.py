# Importing necessary libraries from TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
# Loading the dataset
data_path = 'isolet.csv'
isolet_data = pd.read_csv(data_path)

# Encoding string class labels to integers
label_encoder = LabelEncoder()
isolet_data['class'] = label_encoder.fit_transform(isolet_data['class'])

# Separating feature columns and the target column
X = isolet_data.drop('class', axis=1)
y = isolet_data['class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Setting a random seed to ensure reproducibility of results
tf.random.set_seed(42)

# Building a sequential model in TensorFlow
model = Sequential([
    # First dense layer with 256 neurons, using ReLU (rectified linear unit) as the activation function
    # 'input_shape' specifies the shape of the input data (only required for the first layer)
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    
    # Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 during training
    Dropout(0.5),
    
    # Second dense layer with 128 neurons, also using ReLU as the activation function
    Dense(128, activation='relu'),
    
    # Another dropout layer to further prevent overfitting
    Dropout(0.5),
    
    # Output layer with a number of neurons equal to the number of classes, using softmax for multi-class classification
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compiling the model with the Adam optimizer and setting the learning rate
# 'sparse_categorical_crossentropy' is used as the loss function for multi-class classification tasks
# 'accuracy' metric is used to evaluate the model during training and testing
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model with the training data
# 'epochs' defines the number of times the model will work through the entire training dataset
# 'validation_split' reserves a fraction of the training data for validation to monitor overfitting during training
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, verbose=1)

# Evaluating the model's performance on the test set to check how well it generalizes to new, unseen data
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
