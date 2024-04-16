# Importing necessary libraries from TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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
