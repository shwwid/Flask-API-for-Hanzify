const video = document.getElementById('videoStream');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resultDiv = document.getElementById('result');

let stream = null;
let captureInterval = null;

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.opacity = 1;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    resultDiv.textContent = "Detecting...";

    // Start frame capture every 1 second (adjust as needed)
    captureInterval = setInterval(captureFrame, 1000);
  } catch (err) {
    alert("Could not access the camera. Please allow camera permissions.");
    console.error(err);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  resultDiv.textContent = "Stopped";
  video.style.opacity = 0;
  clearInterval(captureInterval);
}

function captureFrame() {
  // Create a canvas to grab a frame from video
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  // Get base64 of current frame
  const dataUrl = canvas.toDataURL('image/jpeg');

  // Send to server for prediction
  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl })
  })
    .then(res => res.json())
    .then(data => {
      resultDiv.textContent = `${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`;
    })
    .catch(err => {
      console.error('Prediction error:', err);
      resultDiv.textContent = 'Error detecting sign';
    });
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
