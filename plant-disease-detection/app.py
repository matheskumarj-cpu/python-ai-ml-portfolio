from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model/plant_disease_inceptionv3.h5"

CLASS_NAMES = [
    "Banana",
    "Chilli",
    "Corn",
    "Onion",
    "Tomato"
]

IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please select an image."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                error=error
            )

        os.makedirs("static/uploads", exist_ok=True)
        file_path = os.path.join("static/uploads", uploaded_file.filename)
        uploaded_file.save(file_path)

        try:
            img = image.load_img(file_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            probabilities = model.predict(img_array, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))

            prediction = CLASS_NAMES[predicted_index]
            confidence = round(float(probabilities[predicted_index]) * 100, 2)

        except Exception as exc:
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
