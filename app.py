# app.py - Flask Backend for Malaria Detection

import sys
from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime

model = None
MODEL_ERROR = None
MODEL_PATH = None
CLASS_NAMES = ['Parasitized', 'Uninfected']

tf = None
Image = None

# ── Always resolve paths relative to THIS file, not os.getcwd() ──────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n" + "="*70)
print("  🦟 MALARIA DETECTION - CHECKING DEPENDENCIES")
print("="*70)

try:
    import tensorflow as tf
    print("✓ TensorFlow available")
except ImportError as e:
    MODEL_ERROR = f"TensorFlow import failed: {e}"
    print(f"✗ {MODEL_ERROR}")
    tf = None

try:
    from PIL import Image
    print("✓ PIL.Image available")
except ImportError as e:
    if MODEL_ERROR is None:
        MODEL_ERROR = f"Pillow import failed: {e}"
    print(f"✗ {MODEL_ERROR}")
    Image = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

if MODEL_ERROR is None:
    try:
        print(f"✓ Script directory: {BASE_DIR}")

        # Search for .h5 model file in the same folder as app.py
        h5_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.h5')]
        if h5_files:
            MODEL_PATH = os.path.join(BASE_DIR, h5_files[0])
            print(f"✓ Found: {h5_files[0]}")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=['accuracy'])
            print(f"✓ Loaded! Input: {model.input_shape}, Output: {model.output_shape}")
        else:
            MODEL_ERROR = f"No .h5 file found in {BASE_DIR}"
            print(f"✗ {MODEL_ERROR}")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"✗ {MODEL_ERROR}")


# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────
def prepare_image(image_file):
    if model is None or Image is None:
        return None
    try:
        img = Image.open(image_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        input_shape = model.input_shape

        if len(input_shape) == 2:
            img = img.resize((64, 64))
            img_array = np.array(img, dtype='float32') / 255.0
            img_gray = np.mean(img_array, axis=2)
            expected_size = input_shape[1]
            img_flat = img_gray.flatten()
            if len(img_flat) < expected_size:
                img_flat = np.pad(img_flat, (0, expected_size - len(img_flat)))
            else:
                img_flat = img_flat[:expected_size]
            return np.expand_dims(img_flat, axis=0)
        else:
            img_size = (input_shape[1], input_shape[2])
            img = img.resize(img_size)
            img_array = np.array(img, dtype='float32') / 255.0
            return np.expand_dims(img_array, axis=0)

    except Exception as e:
        print(f"Image error: {e}")
        return None


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def home():
    template_path = os.path.join(BASE_DIR, 'templates', 'index.html')
    if os.path.exists(template_path):
        return render_template('index.html')
    return f'''<!DOCTYPE html>
<html><head><title>Malaria Detection</title><style>
body{{font-family:Arial;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
min-height:100vh;padding:20px}}
.container{{max-width:900px;margin:0 auto;background:rgba(0,0,0,0.3);
border-radius:20px;padding:40px;text-align:center}}
.success{{color:#86efac;margin-bottom:10px;font-size:1.1rem}}
.error{{background:#fee2e2;color:#dc2626;padding:10px;border-radius:5px;font-size:1rem}}
</style></head><body>
<div class="container">
<h1>🦟 Malaria Detection System</h1>
<div class="status success">✅ Server Running</div>
<div class="status success">⚡ Port 5000 active</div>
<div class="{'success' if model is not None else 'error'}">
{'✅ MODEL LOADED' if model is not None else '❌ NOT LOADED'}
</div>
<p>{'Ready' if MODEL_ERROR is None else f'Error: {MODEL_ERROR}'}</p>
</div></body></html>'''


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': f'Model not loaded: {MODEL_ERROR}'}), 500

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        img_array = prepare_image(file)
        if img_array is None:
            return jsonify({'success': False, 'error': 'Image processing failed'}), 500

        prediction = model.predict(img_array, verbose=0)
        print(f"Raw prediction: {prediction}, Shape: {prediction.shape}")

        prob = float(prediction[0][0])
        print(f"Probability value: {prob}")

        # prob > 0.5 → model outputs "Uninfected" signal → lower index = Parasitized
        if prob > 0.5:
            pred_class = 0   # Parasitized
            confidence = prob * 100
        else:
            pred_class = 1   # Uninfected
            confidence = (1 - prob) * 100

        result = CLASS_NAMES[pred_class]
        parasitized_prob = (1 - prob) * 100
        uninfected_prob  = prob * 100

        print(f"Result: {result}, Confidence: {confidence:.2f}%")

        return jsonify({
            'success'    : True,
            'prediction' : result,
            'confidence' : round(confidence, 2),
            'details'    : {
                'parasitized_probability': round(parasitized_prob, 2),
                'uninfected_probability' : round(uninfected_prob, 2)
            },
            'timestamp'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status'      : 'running',
        'model_loaded': model is not None,
        'model_error' : MODEL_ERROR
    })


if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'templates'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  {'✅' if model else '❌'} Model: {'LOADED' if model else 'NOT LOADED'}")
    if model:
        print(f"  📥 Input: {model.input_shape}, Output: {model.output_shape}")
    if MODEL_ERROR:
        print(f"  ⚠️  Error: {MODEL_ERROR}")
    print(f"  🌐 Server: http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)