import librosa
import librosa.display
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

file_path = 'isolet.csv'  # Path to the dataset
data = pd.read_csv(file_path, header=None)  # Assuming no header row
print(data.head()) #Inspect the first few rows of the dataframe
print(data.shape)#first few rows of dataframe strucutre
X = data.drop('class', axis=1)  # Features
y = data['class']               # Labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
X_train, X_test, Y_train, Y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

features = data.iloc[:, :-1]  # All rows, all columns except the last one
labels = data.iloc[:, -1]  # All rows, only the last column



scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json

# Load the dataset, skipping the first row if it contains non-relevant information
file_path = 'isolet.csv'
data = pd.read_csv(file_path, header=None, skiprows=1)  # Skipping the first row

# Separating features and labels
features = data.iloc[:, :-1]  # All rows, all columns except the last one for features
labels = data.iloc[:, -1]     # All rows, only the last column for labels

# Encode the labels
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)

# Prepare the data for JSON serialization
data_for_js = {
    "X_train": X_train.tolist(),
    "X_test": X_test.tolist(),
    "Y_train": Y_train.tolist(),
    "Y_test": Y_test.tolist()
}

# Save the processed data to a JSON file for later use in TensorFlow.js
with open('isolet_processed.json', 'w') as json_file:
    json.dump(data_for_js, json_file)

# #pip install ucimlrepo
# from ucimlrepo import fetch_ucirepo 
  
# # fetch dataset 
# isolet = fetch_ucirepo(id=54) 
  
# # data (as pandas dataframes) 
# X = isolet.data.features 
# y = isolet.data.targets 
  
# # metadata 
# print(isolet.metadata) 
  
# # variable information 
# print(isolet.variables) 

def audio_to_spectrogram(file_path, n_mels=128):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB
def normalize(X, mean=-100, std=10):
    return (X - mean) / std
def save_to_json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data.tolist(), f)  # Convert numpy array to list for JSON serialization
data_for_js = {
    "X_train": X_train.tolist(),
    "X_test": X_test.tolist(),
    "Y_train": Y_train.tolist(),
    "Y_test": Y_test.tolist()
}

with open('isolet_processed.json', 'w') as json_file:
    json.dump(data_for_js, json_file)