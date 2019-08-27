let net;
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
let nb_frames = 0

async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Sucessfully loaded model');

    await setupWebcam();

    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = classId => {
        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(webcamElement, 'conv_preds');

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);
    };

    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(1));
    document.getElementById('class-b').addEventListener('click', () => addExample(2));
    document.getElementById('class-c').addEventListener('click', () => addExample(3));

    while (true) {
        if (nb_frames < 100) {
            //first 100 frames used to train the model on no_action
            addExample(0)

        }

        if (classifier.getNumClasses() > 0) {
            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(webcamElement, 'conv_preds');
            // Get the most likely class and confidences from the classifier module.
            const result = await classifier.predictClass(activation, 4);

            const classes = ['No Action', 'A', 'B', 'C'];
            document.getElementById('console').innerText = `
          prediction: ${classes[parseInt(result.label)]}\n
          probability: ${result.confidences[parseInt(result.label)]}
        `;
        }

        nb_frames++;
        await tf.nextFrame();
    }
}

async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia({ video: true },
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener('loadeddata', () => resolve(), false);
                },
                error => reject());
        } else {
            reject();
        }
    });
}

app();