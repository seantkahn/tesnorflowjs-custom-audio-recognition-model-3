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
let examples = [];

function collect(label) {
 if (recognizer.isListening()) {
   return recognizer.stopListening();
 }
 if (label == null) {
   return;
 }
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   examples.push({vals, label});
   document.querySelector('#console').textContent =
       `${examples.length} examples collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}

const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function train() {
 toggleButtons(false);
 // Set the corresponding ys of the labels. - number of classes/labels
 const ys = tf.oneHot(examples.map(e => e.label), 36);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

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

function buildModel() {
 model = tf.sequential();
 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   //change to number of desired classes for model training
   kernelSize: [NUM_FRAMES,  36],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 //change units to number of classes for training
 model.add(tf.layers.dense({units: 36, activation: 'softmax'}));
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

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}

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
   const probs = model.predict(input);
   const predLabel = probs.argMax(1);
   await finish(predLabel);
   tf.dispose([input, probs, predLabel]);
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

async function save () {
    await model.save('downloads://my-model');
}
async function loadNewModel() {
    const model = await tf.loadLayersModel('downloads://my-model');
    //const model = await tf.loadLayersModel('http://localhost:1234/my-model/model.json');
}