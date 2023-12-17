// Define the character set used in your model training
const chars = ['\t', '\n', ' ', '!', '"', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z', '«', '»', 'À', 'Â', 'Ç', 'È', 'É', 'Ê', 'Ô', 'à', 'â', 'ç', 'è', 'é', 'ê', 'î', 'ï', 'ñ', 'ô', 'ù', 'û', 'œ', '–', '’']
const stoi = {};
chars.forEach((char, index) => stoi[char] = index);
const itos = chars;
const vocabSize = 95;
const sequence_length = 64;

const batch_size = 12; // The same as in your Python model



// Encode and decode functions
function encode(str) {
    return str.split('').map(char => stoi[char]);
}

function softmax(arr) {
    const maxLogit = Math.max(...arr);
    const scaled = arr.map(logit => Math.exp(logit - maxLogit));
    const total = scaled.reduce((acc, val) => acc + val, 0);
    return scaled.map(val => val / total);
}
function sampleIndex(probabilities) {
    const rnd = Math.random();
    let cumSum = 0;
    for (let i = 0; i < probabilities.length; i++) {
        cumSum += probabilities[i];
        if (rnd < cumSum) return i;
    }
    return probabilities.length - 1;
}


function decode(outputIndices) {
    let text = '';

    for (let i = 0; i < outputIndices.length; i += vocabSize) {
        const logits = outputIndices.slice(i, i + vocabSize);
        const probabilities = softmax(logits);
        const randomIndex = weightedRandomSelect(probabilities);
        text += itos[randomIndex];
    }

    return text[text.length-1];
}

function weightedRandomSelect(probabilities) {
    let sum = 0;
    const r = Math.random();
    for (let i = 0; i < probabilities.length; i++) {
        sum += probabilities[i];
        if (r <= sum) return i;
    }
    return probabilities.length - 1; // Return the last index in case random number exceeds sum
}


// Global variable for the ONNX session
let session;

// Load the ONNX model using ONNX Runtime Web
async function loadModel() {
    session = await ort.InferenceSession.create("./gpt_language_model.onnx");
}

// Function to generate text based on the user input
async function generateText() {
    if (!session) {
        console.error('Model not loaded yet');
        document.getElementById('outputText').innerText = 'Error: model not loaded yet, please wait'
        return;
    }

    let inputText = document.getElementById('inputText').value;
    let maxLines = document.getElementById('lineInput').value;
    if (!inputText) {
        console.error('Input text is empty');
        document.getElementById('outputText').innerText = 'Error: input text is empty'
        return;
    }
    let encodedInput = encode(inputText);

    if (encodedInput.length === 0) {
        console.error('Encoded input is empty');
        return;
    }

    // Preprocess encodedInput to match the shape [64, 256]
    // This might involve padding or truncating the input

    // Create the input tensor with the correct shape
    // Run the model
    let context = inputText;
    let lines = 0;

    document.getElementById('generateButton').style.visibility = 'hidden';
    try {
        while (lines < maxLines) {

        let paddedContext = context.padEnd(sequence_length, ' '); 
        paddedContext = paddedContext.substring(paddedContext.length - sequence_length); 

        // Repeat the context to match the batch size
        let batchedContext = Array(batch_size).fill(paddedContext).flat().join('');
            
        // Adjust the input tensor shape to [12, 64]
        let inputTensor = new ort.Tensor("int32", encode(batchedContext), [batch_size, sequence_length]);
            
        
        
        
        //let inputTensor = new ort.Tensor("int32", encode(context), [1, context.length]); // Create the input tensor with the correct shape
        const outputMap = await session.run({ 'inputs': inputTensor });
        const outputTensor = outputMap['outputs'];

        const outputIndices = Array.from(outputTensor.data);
        const generatedLetter = decode(outputIndices);
        
        result = inputText+generatedLetter;
        inputText += generatedLetter;

        context += generatedLetter;
        if (context.length >= sequence_length) {
            context = context.slice(1);
        }

        if (generatedLetter == '\n') {
            if (context[context.length-2] == '\n') {
                lines+=1;
            }
        }

        document.getElementById('outputText').innerText = result;
        await new Promise(resolve => setTimeout(resolve, 10));
        }

    } catch (error) {
        console.error('Error during model run:', error);
    }

    document.getElementById('generateButton').style.visibility = 'visible';
}

// Event listener for the Generate button
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('generateButton').addEventListener('click', generateText);
});

document.getElementById('generateButton').style.visibility = 'hidden';
// Load the model immediately when the script is loaded
loadModel();
document.getElementById('generateButton').style.visibility = 'visible';