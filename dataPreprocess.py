#pip install tensorflow librosa numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import librosa
import numpy as np

# Load the pre-trained Speech Commands model
base_model = tf.keras.models.load_model('path/to/speech_commands_model')
base_model.trainable = False  # Freeze the model

# Replace the output layer
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(units=num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on your dataset
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))



#n_mfcc: Number of MFCCs to return.
#max_pad_len: Maximum padding length to ensure uniform feature size across all audio files.
def extract_mfcc(file_path, n_mfcc=13, max_pad_len=174):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs


def create_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Assuming you have 2 classes
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Assuming X_train, X_test, y_train, y_test are prepared
model = create_model(input_shape=(13, 174, 1))  # Update based on your MFCC shape
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save('path/to/save/model.h5')

#pip install tensorflowjs
#tensorflowjs_converter --input_format=keras path/to/save/model.h5 path/to/save/tfjs_model
