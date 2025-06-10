from flask import Flask, render_template, Response, jsonify, request
import base64
import cv2
import numpy as np
import tensorflow as tf
import math
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="words_model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels (only the word part)
with open("words_labels.txt", "r") as f:
    labels = [line.split(' ', 1)[1] for line in f.read().splitlines()]

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Global variable to store last prediction text
last_prediction_text = "No hand detected"

def preprocess_hand(img, bbox):
    x, y, w, h = bbox
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgCrop = img[max(y - offset, 0): y + h + offset, max(x - offset, 0): x + w + offset]

    if imgCrop.size == 0:
        return None

    aspectRatio = h / w
    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wGap + wCal] = imgResize
    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hGap + hCal, :] = imgResize

    return imgWhite

def generate_frames():
    global last_prediction_text
    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img)
        prediction_text = "No hand detected"

        if hands:
            hand = hands[0]
            bbox = hand['bbox']
            imgWhite = preprocess_hand(img, bbox)

            if imgWhite is not None:
                imgInput = cv2.resize(imgWhite, (224, 224))
                imgInput = imgInput.astype(np.float32) / 255.0
                imgInput = np.expand_dims(imgInput, axis=0)

                interpreter.set_tensor(input_details[0]['index'], imgInput)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                index = np.argmax(output_data)
                prediction_text = labels[index]  # Clean label only

        last_prediction_text = prediction_text

        # Put prediction text on the image
        cv2.putText(img, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (124, 138, 255), 3)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/last_prediction')
def last_prediction():
    global last_prediction_text
    return jsonify({'prediction': last_prediction_text})

@app.route('/predict_image', methods=['POST'])
def predict_image():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'message': 'No image provided'}), 400

    try:
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'message': 'Invalid image data'}), 400

        hands, img_hand = detector.findHands(img)
        if not hands:
            return jsonify({'message': 'No hand detected'}), 200

        hand = hands[0]
        bbox = hand['bbox']
        imgWhite = preprocess_hand(img, bbox)
        if imgWhite is None:
            return jsonify({'message': 'Failed to preprocess hand image'}), 200

        imgInput = cv2.resize(imgWhite, (224, 224))
        imgInput = imgInput.astype(np.float32) / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        interpreter.set_tensor(input_details[0]['index'], imgInput)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        index = np.argmax(output_data)
        confidence = output_data[index]

        if confidence > 0.95:
            prediction_text = labels[index]
        else:
            prediction_text = "No confident prediction"
        
        print(f"Prediction: {prediction_text}")
        return jsonify({'message': prediction_text})

    except Exception as e:
        return jsonify({'message': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
