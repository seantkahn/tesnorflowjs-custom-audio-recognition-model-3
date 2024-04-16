#pip install pandas scikit-learn tensorflow

# Importing necessary libraries from TensorFlow
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tensorflow.keras.models import load_model

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
# Save the model
# model.save('isolet_model.h5')  # saves the model in the HDF5 file format

# To save as a TensorFlow SavedModel (better for serving with TensorFlow Serving or TF Lite)
model.save('isolet_model_folder', save_format='tf')

# To load the model

# # Load the model
# model = load_model('isolet_model.h5')  # If saved as an HDF5 file

# # because saved it using the SavedModel format
model = load_model('isolet_model_folder')
# Evaluate the model again
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Reloaded model accuracy: {test_acc}, Test loss: {test_loss}")

# # Make predictions
predictions = model.predict(X_test_scaled)
predicted_classes = tf.argmax(predictions, axis=1)







# Once you have your model saved in the TensorFlow SavedModel format, use the TensorFlow.js converter command-line utility to convert your model to the web format:
#tensorflowjs_converter --input_format=tf_saved_model --output_node_names='Softmax' --saved_model_tags=serve ./isolet_model_folder ./isolet_tfjs_model
# This command will convert the model and create a new directory isolet_tfjs_model with the converted model files. Here are the options used:

# --input_format=tf_saved_model: Specifies the format of the input model.
# --output_node_names='Softmax': This might need to be adjusted based on your model’s output layer name if different.
# --saved_model_tags=serve: Tags used to identify the MetaGraphDef to load from the SavedModel.
# ./isolet_model_folder: The directory of your saved TensorFlow SavedModel.
# ./isolet_tfjs_model: The target directory where the TensorFlow.js model files will be stored.







# Training and Validation Accuracy and Loss:

# Your model reaches a training accuracy of around 97% and a validation accuracy of about 95.51% by the 50th epoch. This suggests that the model is performing well and learning effectively from the training data.
# The training loss decreases steadily, which is a good indicator that the model is learning and optimizing well. The validation loss shows some fluctuations but generally trends downwards, indicating the model's ability to generalize to new, unseen data (though there are signs of potential overfitting as the validation loss starts to increase or fluctuate in later epochs).
# Testing the Model:

# The test accuracy is reported as approximately 95.90%, which is very close to your validation accuracy. This consistency between validation and test accuracy is a good sign and suggests that your model has generalized well and is not merely fitting to peculiarities in the training data.
# Warnings:

# The warning about tf.gfile.Exists being deprecated is due to the TensorFlow version you are using. It's recommending that you use tf.io.gfile.exists instead. This is more of an informational warning and does not affect the execution or performance of your current setup, but it's good practice to update deprecated functions to ensure compatibility with future versions of TensorFlow.
# Next Steps
# Considering the performance of your model, here are a few suggestions for further steps you might take:

# Model Tuning:

# Regularization: If you notice signs of overfitting (where the validation loss increases or fluctuates as training progresses), consider increasing the dropout rate or adding other forms of regularization like L2 regularization.
# Hyperparameter Tuning: Experiment with different learning rates, batch sizes, or numbers of epochs. Additionally, adjusting the architecture (more or fewer layers, different numbers of units per layer) could also yield better results.
# Cross-Validation:

# Implement cross-validation to ensure that your model’s performance is robust and not dependent on the particular split of data used for training and validation.
# Feature Engineering:

# Depending on how the features were initially selected and processed, there might be room for additional feature engineering to help improve the model's ability to learn from the data.
# Update Deprecated Code:

# To avoid future issues with deprecated TensorFlow functions, update your code according to the warnings provided. For example, replacing tf.gfile.Exists with tf.io.gfile.exists.
# Deployment:

# If you're satisfied with the model's performance, consider how you might deploy this model. For a web application, converting it to TensorFlow.js as previously discussed could be a great option.
# Continuous Monitoring:

# Once deployed, continuously monitor the model's performance to catch any degradation or changes in data that might affect its accuracy.