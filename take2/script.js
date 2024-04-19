// python -m http.server 8000
// http://127.0.0.1:8000
//initializing the model
    //recognizer holds to model
    //app() function is called to create the model - An asynchronous function that initializes the speech recognizer with the BROWSER_FFT model (optimized for browsers) and then calls buildModel() to prepare the custom model for training.
//collecting the audio samples
    //NUM_FRAMES: Defines the number of frames to consider for a single audio sample. Each frame represents ~23ms of audio.
    //examples: An array to store the collected audio samples and their corresponding labels.
    //collect(label): A function to collect audio samples. If the recognizer is currently listening, it stops it. Otherwise, it starts listening and pushes collected audio samples (normalized spectrogram data) along with their labels to the examples array.
//normalizing the audio data
    //normalize(x): A function to normalize the audio data by subtracting the mean and dividing by the standard deviation. - Normalizes the audio sample data using a predefined mean and standard deviation to ensure the model receives data within a suitable range.
//training the model
    //INPUT_SHAPE: Specifies the input shape of the model, derived from the number of frames and the dimensions of the spectrogram.
    //model: A variable to store the TensorFlow.js model.
    //train(): An asynchronous function that prepares the data (as tensors) for training, including one-hot encoding the labels, and then trains the model using the collected samples.
//building the model
    //buildModel(): A function to construct the TensorFlow.js model architecture, including depthwise convolutional and dense layers. - Defines the model architecture using TensorFlow.js layers, including depthwise convolutional and dense layers.
    //buildModel(): Constructs a sequential neural network model with a specific architecture suitable for processing the audio data. This includes convolutional and dense layers, with the final output layer having as many units as there are labels to classify.
//Utility Functions
    //toggleButtons(enable): Enables or disables buttons on the web page, typically used during training to prevent additional input.
    //flatten(tensors): A utility function to flatten an array of tensors into a single Float32Array, useful for preprocessing data before training.
    //labels: An array of strings representing the commands or labels that the model can recognize.
//preditiction and listening
    //finish(labelTensor): An asynchronous function that extracts the predicted label from the model's output tensor and updates the web page with the corresponding label.
    //listen(): A function that toggles between starting and stopping the speech recognition process. When listening, it processes the audio data, predicts the label using the model, and updates the web page with the result.
    //finish(labelTensor): Processes the prediction result, displaying the recognized command based on the model's output.
    //listen(): Toggles the listening state of the recognizer and makes predictions based on the incoming audio, using the trained model.
//saving the model
    //save(): An asynchronous function that loads the trained model from the server and stores it in a variable for future use.
    //save(): Demonstrates how to load a model from a specified URL. This might be intended for loading a pre-trained model, although in this context, it seems more appropriate as a placeholder or example of how to implement model loading.
//2 5 6 8 9 11

let recognizer;
async function app() {
 recognizer = speechCommands.create('BROWSER_FFT');
 await recognizer.ensureModelLoaded();
 // Add this line.
 buildModel();
}

app();

// One frame is ~23ms of audio.
const NUM_FRAMES = 6;
//each training example will have 2 fields:
//label****: 0, 1, and 2 for "Left", "Right" and "Noise" respectively.
//vals****: 696 numbers holding the frequency information (spectrogram)
//and we store all data in the examples variable:
let examples = [];
//collect associates a label with the output of recognizer.lsiten, which provides a raw spectrogram because includespectrogram is set to true above. 
//collect creates training examples for the model
function collect(label) {
 if (recognizer.isListening()) {
   return recognizer.stopListening();
 }
 if (label == null) {
   return;
 }
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
  //To avoid numerical issues, we normalize the data to have an average of 0 and a standard deviation of 1. In this case, the spectrogram values are usually large negative numbers around -100 and deviation of 10: 
  let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   examples.push({vals, label});
  //  document.querySelector('#console').textContent =
  //      `${examples.length} examples collected`;
  labelCounts[label] += 1;  // Increment the count for the label
  updateExampleCountUI();  // Update the UI

 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}
//To avoid numerical issues, we normalize the data to have an average of 0 and a standard deviation of 1. 
//In this case, the spectrogram values are usually large negative numbers around -100 and deviation of 10:

function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}
//The input shape of the model is [NUM_FRAMES, 232, 1] where each frame is 23ms of audio containing 232 numbers that correspond to different frequencies (232 was chosen because it is the amount of frequency buckets needed to capture the human voice). 
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;
//The model is a convolutional neural network with a depthwise convolutional layer, a max pooling layer, and a dense layer. 
    //The model is compiled with the Adam optimizer and the categorical crossentropy loss function. The training process is done in batches of 16 examples for 10 epochs. 
    //The training process is asynchronous and the UI is updated with the accuracy and epoch number at the end of each epoch.
//we are doing two things: buildModel() defines the model architecture and train() trains the model using the collected data. 

async function train() {
 toggleButtons(false);
 // Set the corresponding ys of the labels. - number of classes/labels
 const ys = tf.oneHot(examples.map(e => e.label), 36);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);
 //The training goes 10 times (epochs) over the data using a batch size of 16 (processing 16 examples at a time) and shows the current accuracy in the UI:
 await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
}

//The input shape of the model is [NUM_FRAMES, 232, 1] where each frame is 23ms of audio containing 232 numbers that correspond to different frequencies (232 was chosen because it is the amount of frequency buckets needed to capture the human voice). 
//In this codelab, we are using samples that are 3 frames long (~70ms samples) since we are making sounds instead of speaking whole words to control the slider.
//The model has 4 layers: a convolutional layer that processes the audio data (represented as a spectrogram), a max pool layer, a flatten layer, and a dense layer that maps to the 3 actions:
function buildModel() {
 model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   //change to number of desired classes for model training? - NO
   kernelSize: [NUM_FRAMES,  3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 //change units to number of classes for training
 model.add(tf.layers.dense({units: 36, activation: 'softmax'}));
  //We compile our model to get it ready for training:
  //We use the Adam optimizer and the categorical crossentropy loss function, which is suitable for multiclass classification problems.
 //We use the Adam optimizer, a common optimizer used in deep learning, and categoricalCrossEntropy for loss, the standard loss function used for classification. 
//In short, it measures how far the predicted probabilities (one probability per class) are from having 100% probability in the true class, and 0% probability for all the other classes. 
//We also provide accuracy as a metric to monitor, which will give us the percentage of examples the model gets correct after each epoch of training.
  const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}
//converts an array of tensors into a single Float32Array
function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}
//Data class labels
var labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", 
    "I", "J", "K", "L", "M", "N", "O", "P", 
    "Q", "R", "S", "T", "U", "V", "W", "X", 
    "Y", "Z", "Apple", "Bird", "Boat", "Butterfly", 
    "Car", "Dog", "Cat", "Horse", "Train", "Noise"
  ];
async function finish(labelTensor) {
 const label = (await labelTensor.data())[0];
 document.getElementById('console').textContent = labels[label];
} 
//listen() listens to the microphone and makes real time predictions. The code is very similar to the collect() method, which normalizes the raw spectrogram and drops all but the last NUM_FRAMES frames. The only difference is that we also call the trained model to get a prediction:
//The output of model.predict(input)is a Tensor of shape [1, numClasses] representing a probability distribution over the number of classes. More simply, this is just a set of confidences for each of the possible output classes which sum to 1. The Tensor has an outer dimension of 1 because that is the size of the batch (a single example).
//To convert the probability distribution to a single integer representing the most likely class, we call probs.argMax(1)which returns the class index with the highest probability. We pass a "1" as the axis parameter because we want to compute the argMax over the last dimension, numClasses.
function listen() {
 if (recognizer.isListening()) {
   recognizer.stopListening();
   toggleButtons(true);
   document.getElementById('listen').textContent = 'Listen';
   return;
 }
 toggleButtons(false);
 document.getElementById('listen').textContent = 'Stop';
 document.getElementById('listen').disabled = false;

 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
   //The model.predict() method returns a tensor with the probabilities of each class. We use argMax(1) to get the index of the class with the highest probability. We then call moveSlider() to update the slider position based on the predicted class.
   const probs = model.predict(input);
   const predLabel = probs.argMax(1);
   //print model prediction
   await finish(predLabel);
   //Disposing tensors is important to free up memory. TensorFlow.js does not automatically clean up memory, so it's important to manually dispose of tensors when they are no longer needed.
//To clean up GPU memory it's important for us to manually call tf.dispose() on output Tensors. The alternative to manual tf.dispose() is wrapping function calls in a tf.tidy(), but this cannot be used with async functions.
   tf.dispose([input, probs, predLabel]);
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}
// document.getElementById('load-files').addEventListener('click', async () => {
//   let files = document.getElementById('file-input').files;
//   for (let file of files) {
//       let audioBuffer = await file.arrayBuffer();
//       let tensor = await convertToTensor(audioBuffer);
//       label = 
//       examples.push({vals: tensor, label: 0}); // Adjust the label as needed
//   }
// });
 
const labelIds = [
  "A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", 
  "I1", "J1", "K1", "L1", "M1", "N1", "O1", "P1", 
  "Q1", "R1", "S1", "T1", "U1", "V1", "W1", "X1", 
  "Y1", "Z1", "Apple1", "Bird1", "Boat1", "Butterfly1", 
  "Car1", "Dog1", "Cat1", "Horse1", "Train1", "Noise1"
];
// async function loadLabel(numericLabel, files){
//   const elementId = labels[numericLabel]; // Get the correct ID from the array
//   let files = document.getElementById(elementId).files;
//   if (files.length === 0) {
//     console.log("No files selected.");
//     return;
//   }
//   for (let file of files) {
//     let audioBuffer = await file.arrayBuffer();
//     let tensor = await convertToTensor(audioBuffer);
//     examples.push({vals: tensor, label: numericLabel}); // Adjust the label as needed
//     labelCounts[numericLabel] += 1;  // Increment the count for the label
//   }
//   updateExampleCountUI();  // Update the UI after all files are processed
// }
async function loadLabel(numericLabel, files) {
  for (let file of files) {
      let audioBuffer = await file.arrayBuffer();
      let tensor = await convertToTensor(audioBuffer);
      examples.push({vals: tensor, label: numericLabel}); // Add to your training examples
  }
  updateExampleCountUI(); //update UI to show how many examples have been loaded
}

async function calculateNormalizationParameters() {//call before training to compute mean and deviation from laoded data
  let allData = [];
  // Assuming 'examples' is filled with some initial data to calculate stats
  for (let example of examples) {
      allData = allData.concat(Array.from(example.vals));
  }
  const dataTensor = tf.tensor1d(allData);
  globalMean = dataTensor.mean().dataSync()[0];
  globalStd = dataTensor.std().dataSync()[0];
  dataTensor.dispose();
  updateExampleCountUI();
}
let labelCounts = new Array(36).fill(0);  // Assuming you have 36 labels, from 0 to 35

function updateExampleCountUI() {
  const consoleDiv = document.getElementById('console');
  consoleDiv.innerHTML = '';  // Clear previous contents
      // Display normalization parameters
      consoleDiv.innerHTML += `<strong>Normalization Parameters:</strong><br>`;
      consoleDiv.innerHTML += `Mean: ${globalMean.toFixed(2)}, Standard Deviation: ${globalStd.toFixed(2)}<br><br>`;
  
      // Display example counts
      consoleDiv.innerHTML += `<strong>Label Counts:</strong><br>`;
  labelCounts.forEach((count, index) => {
      if (count > 0) {  // Only display labels with one or more examples
          consoleDiv.innerHTML += `Label ${labels[index]}: ${count} examples<br>`;
      }
  });
}


async function convertToTensor(audioBuffer) {
  // Decode the audio to a tensor
  const waveform = await tf.audio.decodeWav(new Uint8Array(audioBuffer), {desiredSamples: 44100});
  // You might need to adjust the number of samples or the way you handle the audio
  
  // Convert the waveform to a spectrogram
  const spectrogram = tf.signal.stft(waveform, 1024, 256);
  const magnitudeSpectrogram = tf.abs(spectrogram).sum(2);
  const normalizedSpectrogram = normalize(magnitudeSpectrogram); // Use your existing normalize function
  return normalizedSpectrogram;
}


async function save () {
    await model.save('downloads://my-model');
}
async function loadNewModel() {
      // TensorFlow.js expects an object mapping names to URLs for the weight files
  const model = await tf.loadLayersModel('"C:\Users\Seank\OneDrive\Desktop\Models\my-model.json"');
    await model.ensureModelLoaded();
      // TensorFlow.js expects an object mapping names to URLs for the weight files
    buildModel();
    //const model = await tf.loadLayersModel('http://localhost:1234/my-model/model.json');
}