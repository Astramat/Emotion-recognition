let model;
const emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'];
const audioFiles = {
    'Happy': 'musics/happy.mp3',
    'Surprise': 'musics/surprise.mp3',
    'Angry': 'musics/angry.mp3',
    'Sad': 'musics/sad.mp3',
    'Disgust': 'musics/disgust.mp3',
    'Fear': 'musics/fear.mp3',
    'Neutral': 'musics/neutral.mp3'
};
let currentAudio;
// Charger le modèle TensorFlow.js
async function loadModel() {
    model = await tf.loadLayersModel('./model_js/model.json');
    console.log('Model loaded!');
}

// Initialiser la webcam
async function setupCamera() {
    const video = document.getElementById('video');
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

// Prétraitement de l'image
function preprocessImage(video) {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 48;
    canvas.height = 48;

    // Dessiner la vidéo sur le canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Obtenir les données de l'image
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Convertir en niveau de gris
    let grayImageData = tf.tidy(() => {
        let imgTensor = tf.browser.fromPixels(imageData);
        imgTensor = tf.image.resizeBilinear(imgTensor, [48, 48]);
        imgTensor = imgTensor.mean(2).expandDims(-1);
        imgTensor = imgTensor.expandDims(0).div(255.0);
        return imgTensor;
    });

    return grayImageData;
}

// Prédiction en temps réel
async function predictEmotion(video) {
    const processedImage = preprocessImage(video);
    const prediction = model.predict(processedImage);
    const emotionIndex = prediction.argMax(-1).dataSync()[0];
    processedImage.dispose();
    return emotions[emotionIndex];
}

function playMusic(emotion) {
    if (currentAudio && currentAudio._src === audioFiles[emotion]) {
        return;  // Si la musique actuelle correspond à la musique précédente, ne pas jouer de nouveau
    }

    // Arrêter la musique précédente si elle est en cours et la nouvelle n'est pas identique
    if (currentAudio) {
        currentAudio.stop();
    }

    currentAudio = new Howl({
        src: [audioFiles[emotion]],
        html5: true,
        seekable: true,
        onend: () => {
            currentAudio = null;
        }
    });

    currentAudio.play();
}


// Démarrer la reconnaissance en temps réel
async function startEmotionRecognition() {
    const video = await setupCamera();
    await loadModel();

    const emotionText = document.getElementById('emotion');

    setInterval(async () => {
        const emotion = await predictEmotion(video);
        console.log(emotion);
        emotionText.textContent = emotion;

        playMusic(emotion);
    }, 200);
}

// Lancer l'application
startEmotionRecognition();
