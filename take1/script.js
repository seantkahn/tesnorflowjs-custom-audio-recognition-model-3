const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}


let recognizer;
//training code below - collect and normalize data for training
// One frame is ~23ms of audio.
//each training example will have 2 fields:
//label****: 0, 1, and 2 for "Left", "Right" and "Noise" respectively.
//vals****: 696 numbers holding the frequency information (spectrogram)
//and we store all data in the examples variable:
const NUM_FRAMES = 3;
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
   //And to avoid numerical issues, we normalize the data to have an average of 0 and a standard deviation of 1. In this case, the spectrogram values are usually large negative numbers around -100 and deviation of 10:
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   examples.push({vals, label});
   console.log(examples);
   document.querySelector('#console').textContent =
       `${examples.length} examples collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

window.collect = collect;
async function saveModel() {
    await model.save('downloads://my-model');
}

async function loadNewModel() {
    const model = await tf.loadLayersModel('http://localhost:1234/my-model/model.json');
}
async function loadModelFromFiles() {
    const jsonUpload = document.getElementById('upload-json');
    const weightsUpload = document.getElementById('upload-weights');
  
    if (jsonUpload.files.length > 0 && weightsUpload.files.length > 0) {
      const modelURL = URL.createObjectURL(jsonUpload.files[0]);
      const weightsURLs = Array.from(weightsUpload.files).map(f => URL.createObjectURL(f));
      
      // TensorFlow.js expects an object mapping names to URLs for the weight files
      const model = await tf.loadLayersModel(tf.io.browserFiles([jsonUpload.files[0], ...weightsUpload.files]));
      console.log('Model loaded successfully');
    }
  }
  
function updateModelForNewLabels() {
    const numLabels = labels.length;
    model.layers[model.layers.length - 1].units = numLabels;
    // Other necessary updates to the model, then recompile
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
}

// Function to add a new label
function addNewLabel(labelName) {
    // Add new label to an array of labels
    if (!labels.includes(labelName)) {
        labels.push(labelName);
        // Update the model's last layer to accommodate the new number of classes
        updateModelForNewLabels();
    }
}


//And to avoid numerical issues, we normalize the data to have an average of 0 and a standard deviation of 1. In this case, the spectrogram values are usually large negative numbers around -100 and deviation of 10:

function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}


function toggleButtons(enable) {
    document.querySelectorAll('button').forEach(b => b.disabled = !enable);
   }


//The input shape of the model is [NUM_FRAMES, 232, 1] where each frame is 23ms of audio containing 232 numbers that correspond to different frequencies (232 was chosen because it is the amount of frequency buckets needed to capture the human voice). 
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;
//The model is a convolutional neural network with a depthwise convolutional layer, a max pooling layer, and a dense layer. 
    //The model is compiled with the Adam optimizer and the categorical crossentropy loss function. The training process is done in batches of 16 examples for 10 epochs. 
    //The training process is asynchronous and the UI is updated with the accuracy and epoch number at the end of each epoch.
//At a high level we are doing two things: buildModel() defines the model architecture and train() trains the model using the collected data. 
async function train() {
 toggleButtons(false);
 //change3 second onehot parameter integer to number of classes that are being trained
 const ys = tf.oneHot(examples.map(e => e.label), 3);
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

   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));
 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 //change3 second parameter to number of classes that are being trained
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));

 //We compile our model to get it ready for training:
 const optimizer = tf.train.adam(0.01);
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}
//We use the Adam optimizer, a common optimizer used in deep learning, and categoricalCrossEntropy for loss, the standard loss function used for classification. 
//In short, it measures how far the predicted probabilities (one probability per class) are from having 100% probability in the true class, and 0% probability for all the other classes. 
//We also provide accuracy as a metric to monitor, which will give us the percentage of examples the model gets correct after each epoch of training.


function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}

//moveSlider() decreases the value of the slider if the label is 0 ("Left") , increases it if the label is 1 ("Right") and ignores if the label is 2 ("Noise").
async function moveSlider(labelTensor) {
    const label = (await labelTensor.data())[0];
    document.getElementById('console').textContent = label;
    if (label == 2) {
      return;
    }
    let delta = 0.1;
    const prevValue = +document.getElementById('output').value;
    document.getElementById('output').value =
        prevValue + (label === 0 ? -delta : delta);
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
      //moveSlider() decreases the value of the slider if the label is 0 ("Left") , increases it if the label is 1 ("Right") and ignores if the label is 2 ("Noise").
      await moveSlider(predLabel);
//Disposing tensors is important to free up memory. TensorFlow.js does not automatically clean up memory, so it's important to manually dispose of tensors when they are no longer needed.
//To clean up GPU memory it's important for us to manually call tf.dispose() on output Tensors. The alternative to manual tf.dispose() is wrapping function calls in a tf.tidy(), but this cannot be used with async functions.
      tf.dispose([input, probs, predLabel]);
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
}


window.listen = listen;
window.train = train;
function predictWord() {
 // Array of words that the recognizer is trained to recognize.
 const words = recognizer.wordLabels();
 recognizer.listen(({scores}) => {
   // Turn scores into a list of (score,word) pairs.
   scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
   // Find the most probable word.
   scores.sort((s1, s2) => s2.score - s1.score);
   document.querySelector('#console').textContent = scores[0].word;
 }, {probabilityThreshold: 0.75}); // Probability threshold is the conidence level at which a result is returned by the mdoel

}


async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    //predictWord();
   }
app();



   //todo
  // when training the model, is the model creating new classes based on the buttons that call the collect function? where does this training data go, and how do i know what the models classes are and how to add more? If I were to end the session, would the model cease to exist and I would have to start from scratch training again?

   //save model and load up model in order to continue training and using it in the future
    //add more classes to model