# from flask import Flask, render_template
from flask import Flask, Response, render_template
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

camera = cv2.VideoCapture(0)

# Load DEEPFACE model
model = DeepFace.build_model('Emotion')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Resize frame
        resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Preprocess the image for DEEPFACE
        img = gray_frame.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Predict emotions using DEEPFACE
        preds = model.predict(img)
        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (0, 0), (200, 30), (0, 0, 0), -1)
        cv2.putText(frame, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/',methods=["GET"])
def home():
    return render_template('login.html')

@app.route('/About',methods=["GET"])
def about():
    return render_template('about.html')

@app.route('/Index',methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/test',methods=["GET"])
def test():
    return render_template('test.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(debug=True)
