# 🧠 FLASK API FOR HANZIFY - REAL-TIME ASL DETECTION AND TRANSLATION
The Flask API is a critical component of the Hanzify system, acting as the bridge between the front-end application and the underlying machine learning model. It enables real-time video streaming, gesture detection, and on-the-fly translation of American Sign Language (ASL) gestures into text and speech.

# 🔧 Key Responsibilities
Serve the custom-trained MobileNetV2-based deep learning model for ASL recognition.
- Accept and process live video frames from the client app (e.g., Flutter).
- Perform gesture prediction on the received frames.
- Return real-time classification results (translated text) to the client.
- Enable text-to-speech (TTS) support by returning predictions for vocalization.
- Optionally support multiple languages using translation APIs (e.g., Assamese, Manipuri, Hindi).

# How It Works
- Live Video Feed: The client sends video frames or MJPEG stream to the Flask server.
- Frame Processing: Frames are preprocessed and passed through the ML model.
- ASL Detection: The model detects the ASL gesture and outputs the corresponding label.
- Text Output: The prediction is sent back to the app for display and/or voice playback.
