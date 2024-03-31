import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import pickle
from PIL import Image
from io import BytesIO
import cv2
import base64
import keras

camera = cv2.VideoCapture(0)
app = Flask(__name__)
model = keras.load("Saved_model/model")


@app.route("/")
def home():
    return render_template("MF.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        success, img_gen = camera.read()

        # img = Image.fromarray(img_gen, "RGB")
        cv2.imwrite("image.jpg", img_gen)
        img_gen = Image.open(
            "C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Male_or_female/image.jpg"
        )
        img_gen = img_gen.resize((224, 224))
        img_gen = np.ravel(img_gen)
        prediction = model.predict([img_gen])

        output = ["Male!", "Female!"][prediction[0]]

        return render_template("MF.html", prediction_text=f"{output}")

    except Exception as e:
        print(e)
        return render_template("MF.html")


@app.route("/results", methods=["POST"])
def results():

    data = request.get_json(force=True)

    img_gen = Image.open(
        "C:/Users/anshg/OneDrive/Desktop/Python shit/ML/Male_or_female/image.jpg"
    )
    img_gen = img_gen.resize((32, 32))
    img_gen = np.ravel(img_gen)
    prediction = model.predict([img_gen])

    output = ["Male!", "Female!"][prediction[0]]
    return jsonify(output)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )  # concat frame one by one and show result


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
