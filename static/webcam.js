const video = document.getElementById('webcam');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

function captureAndPredict() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const dataUrl = canvas.toDataURL('image/jpeg');

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataUrl })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById('result').innerText =
      `Prediction: ${data.prediction}, Confidence: ${data.confidence}`;
  });
}
