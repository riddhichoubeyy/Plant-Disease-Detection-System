from flask import Flask, render_template, request, jsonify
import json
import os
import base64
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ─── Load Treatment JSON ───────────────────────────────────────────────────────
with open('treatments.json', 'r') as f:
    TREATMENTS = json.load(f)

# ─── Load Class Names (shared by EfficientNetB0 & MobileNetV2 — same dataset) ──
with open('class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

# ─── Load Models ────────────────────────────────────────────────────────────────
# Both models were trained on the same 38-class PlantVillage (color) dataset,
# so they share CLASS_NAMES above. ResNet50 is intentionally NOT included here —
# it was trained on a different, smaller dataset (only 15 classes: Pepper/Potato/
# Tomato), so its output indices don't correspond to CLASS_NAMES and it can't be
# swapped in through the same dropdown without a separate class list + UI path.
import tensorflow as tf

MODELS = {
    "efficientnet": {
        "name": "EfficientNetB0",
        "accuracy": "98%",
        "model": tf.keras.models.load_model("models/efficientnetb0.keras"),
    },
    "mobilenet": {
        "name": "MobileNetV2",
        "accuracy": "95%",
        "model": tf.keras.models.load_model("models/mobilenetv2.keras"),
    },
}

DEFAULT_MODEL_KEY = "efficientnet"


# ─── Predict Function ──────────────────────────────────────────────────────────
def predict_disease(image: Image.Image, model_key: str) -> dict:
    model_entry = MODELS.get(model_key, MODELS[DEFAULT_MODEL_KEY])
    model = model_entry["model"]

    img = image.resize((224, 224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    return {
        "label": CLASS_NAMES[idx],
        "confidence": float(preds[0][idx]),
        "model_used": model_entry["name"],
    }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    # Pass available models to the template so the dropdown can be built server-side
    model_options = [
        {"key": key, "name": entry["name"], "accuracy": entry["accuracy"]}
        for key, entry in MODELS.items()
    ]
    return render_template('index.html', model_options=model_options, default_model=DEFAULT_MODEL_KEY)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Which model did the user pick in the dropdown?
    model_key = request.form.get('model', DEFAULT_MODEL_KEY)
    if model_key not in MODELS:
        return jsonify({'error': f'Unknown model "{model_key}"'}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        result = predict_disease(image, model_key)
        label = result['label']
        confidence = result['confidence']

        treatment_info = TREATMENTS.get(label, {
            "disease_name": label,
            "description": "No description available.",
            "symptoms": [],
            "treatment": [],
            "prevention": []
        })

        buffered = io.BytesIO()
        image.thumbnail((600, 600))
        image.save(buffered, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'label': label,
            'confidence': round(confidence * 100, 2),
            'model_used': result['model_used'],
            'treatment': treatment_info,
            'image': img_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)