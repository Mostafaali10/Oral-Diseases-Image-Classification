from pathlib import Path
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "resnet50_hparam_best.keras"
UPLOADS = Path(__file__).parent / "uploads"
UPLOADS.mkdir(exist_ok=True)

CLASS_NAMES = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth_Ulcer",
    "Tooth_Discoloration",
]

app = Flask(__name__)
model = keras.models.load_model(MODEL_PATH)
print("Loaded", MODEL_PATH.name)


def preprocess(path):
    img = keras.utils.load_img(path, target_size=(224, 224))
    arr = keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def predict_file(path):
    probs = model.predict(preprocess(path), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return {
        "class": CLASS_NAMES[int(order[0])],
        "confidence": round(float(probs[order[0]] * 100), 2),
        "all_probs": {
            CLASS_NAMES[int(i)]: round(float(probs[i] * 100), 2) for i in order
        },
    }


@app.route("/uploads/<name>")
def uploaded(name):
    return send_from_directory(UPLOADS, secure_filename(name))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return render_template("index.html", error="اختر صورة أولاً")
        dest = UPLOADS / secure_filename(file.filename)
        file.save(dest)
        return render_template(
            "index.html",
            result=predict_file(dest),
            filename=dest.name,
        )
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file"}), 400
    dest = UPLOADS / secure_filename(file.filename)
    file.save(dest)
    out = predict_file(dest)
    return jsonify({
        "predicted_class": out["class"],
        "confidence": out["confidence"],
        "probabilities": out["all_probs"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
