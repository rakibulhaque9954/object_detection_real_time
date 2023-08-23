let video, canvas, ctx;
let cascadeClassifier;

function onOpenCvReady() {
    video = document.getElementById('webcam');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            video.srcObject = stream;
        })
        .catch(function (error) {
            console.error('Error accessing the webcam:', error);
        });
    } else {
        console.error('getUserMedia is not supported by this browser.');
    }

    const modelPath = 'haarcascade_frontalface_default.xml'; // Adjust the path to your model file
    cv.onRuntimeInitialized = () => {
        cascadeClassifier = new cv.CascadeClassifier();
        cascadeClassifier.load(modelPath);
        detectObjects();
    };
}

function detectObjects() {
    if (!video.paused && !video.ended) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const src = cv.imread(canvas);
        const gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        const faces = new cv.RectVector();
        cascadeClassifier.detectMultiScale(gray, faces);

        for (let i = 0; i < faces.size(); ++i) {
            const face = faces.get(i);
            const point1 = new cv.Point(face.x, face.y);
            const point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
        }

        cv.imshow(canvas, src);
        src.delete();
        gray.delete();
        faces.delete();

        requestAnimationFrame(detectObjects);
    }
    else {
        // Handle video stream end or pause
        // You might want to implement some logic here to stop the detection loop
    }
}
