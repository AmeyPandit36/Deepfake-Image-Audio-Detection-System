import os
import io
import json
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

# ─── Configuration ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ─── Model Registry ─────────────────────────────────────────────
IMAGE_MODELS = {
    "custom_cnn": {
        "id": "custom_cnn",
        "name": "Custom CNN",
        "subtitle": "Without Data Augmentation",
        "description": "A custom 3-block CNN trained on 100K real/fake face images. Achieves the best overall performance.",
        "architecture": "3× Conv Blocks → GlobalAvgPool (512-dim) → Dense → Sigmoid",
        "input_shape": "224 × 224 × 3",
        "accuracy": 86,
        "roc_auc": 0.985,
        "ap_score": 0.984,
        "precision_real": 0.79,
        "recall_real": 0.99,
        "precision_fake": 0.99,
        "recall_fake": 0.74,
        "f1_real": 0.88,
        "f1_fake": 0.84,
        "file": "custom_model.h5",
        "type": "keras",
        "layer_name": "global_average_pooling2d",
        "category": "image"
    },
    "custom_cnn_aug": {
        "id": "custom_cnn_aug",
        "name": "Custom CNN (Augmented)",
        "subtitle": "With Data Augmentation",
        "description": "Same CNN architecture trained with flips, rotations, zooms. Improves real-world generalization.",
        "architecture": "Augment → 3× Conv Blocks → GlobalAvgPool (512-dim) → Dense → Sigmoid",
        "input_shape": "224 × 224 × 3",
        "accuracy": 73,
        "roc_auc": 0.937,
        "ap_score": 0.933,
        "precision_real": 0.65,
        "recall_real": 0.98,
        "precision_fake": 0.96,
        "recall_fake": 0.47,
        "f1_real": 0.78,
        "f1_fake": 0.63,
        "file": "custom_augmented_model.h5",
        "type": "keras",
        "layer_name": "global_average_pooling2d",
        "category": "image"
    },
    "resnet_v1": {
        "id": "resnet_v1",
        "name": "ResNet-50 Classifier",
        "subtitle": "PyTorch Vision Model",
        "description": "A fine-tuned ResNet-50 architecture modified with a custom linear head. Learns deep spatial features for deepfake detection.",
        "architecture": "ResNet-50 Base → Linear(2048, 512) → ReLU → Dropout → Linear(512, 1)",
        "input_shape": "224 × 224 × 3",
        "accuracy": 89,
        "roc_auc": 0.950,
        "ap_score": 0.942,
        "precision_real": 0.85,
        "recall_real": 0.92,
        "precision_fake": 0.91,
        "recall_fake": 0.83,
        "f1_real": 0.88,
        "f1_fake": 0.87,
        "file": "resnet_image_classifier_v1.pth",
        "type": "pytorch",
        "category": "image"
    },
}

AUDIO_MODELS = {
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest",
        "subtitle": "MFCC Feature Classifier",
        "description": "A Random Forest classifier trained on Mel-Frequency Cepstral Coefficient (MFCC) features extracted from audio signals for deepfake audio detection.",
        "architecture": "Audio → MFCC Features → Random Forest",
        "input_shape": "Audio waveform",
        "accuracy": 82,
        "roc_auc": 0.91,
        "ap_score": 0.91,
        "precision_real": 0.80,
        "recall_real": 0.85,
        "precision_fake": 0.84,
        "recall_fake": 0.79,
        "f1_real": 0.82,
        "f1_fake": 0.81,
        "file": "random_forest_model.pkl",
        "type": "sklearn",
        "category": "audio"
    },
}

ALL_MODELS = {**IMAGE_MODELS, **AUDIO_MODELS}

# ─── Lazy-Loaded Models ─────────────────────────────────────────
_models_cache = {}

def get_model(model_id):
    if model_id in _models_cache:
        return _models_cache[model_id]
    info = ALL_MODELS[model_id]
    path = os.path.join(MODELS_DIR, info['file'])
    if info['type'] == 'keras':
        from tensorflow.keras.models import load_model
        model = load_model(path, compile=False)
        _models_cache[model_id] = model
    elif info['type'] == 'sklearn':
        import joblib
        model = joblib.load(path)
        _models_cache[model_id] = model
    elif info['type'] == 'pytorch':
        import torch
        import torch.nn as nn
        import torchvision.models as custom_models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = custom_models.resnet50(weights=None)
        
        # Match the architecture found in the checkpoint
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        _models_cache[model_id] = model
        
    return _models_cache[model_id]

# ─── Image Preprocessing ─────────────────────────────────────────
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_image_pt(image_bytes):
    from torchvision import transforms
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# ─── Audio Feature Extraction (MFCC) ─────────────────────────────
def extract_audio_features(audio_bytes, filename):
    """Extract MFCC features from audio for RF classification."""
    try:
        import librosa
        import soundfile as sf
        import tempfile

        # Write bytes to temp file
        ext = filename.rsplit('.', 1)[-1].lower()
        with tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=30)
            # Extract MFCC features (24 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
            # Statistics over time: mean + std = 48 features
            features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            return features.reshape(1, -1)
        finally:
            os.unlink(tmp_path)
    except ImportError:
        raise RuntimeError("librosa not installed. Run: pip install librosa soundfile")

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_audio(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

# ─── Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models_page():
    return render_template('models.html',
                           image_models=list(IMAGE_MODELS.values()),
                           audio_models=list(AUDIO_MODELS.values()))

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

# ─── API: Model Info ─────────────────────────────────────────────
@app.route('/api/models')
def api_models():
    return jsonify({
        'image': list(IMAGE_MODELS.values()),
        'audio': list(AUDIO_MODELS.values())
    })

# ─── API: Image Predict ──────────────────────────────────────────
@app.route('/api/predict/image', methods=['POST'])
def api_predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file.filename or not allowed_image(file.filename):
        return jsonify({'error': 'Invalid file. Upload PNG, JPG, JPEG, WEBP, or BMP.'}), 400

    image_bytes = file.read()
    img_array = preprocess_image(image_bytes)
    results = []

    for model_id, info in IMAGE_MODELS.items():
        try:
            model = get_model(model_id)
            if info['type'] == 'pytorch':
                import torch
                img_pt = preprocess_image_pt(image_bytes)
                device = next(model.parameters()).device
                img_pt = img_pt.to(device)
                with torch.no_grad():
                    out = model(img_pt)
                    prob = float(torch.sigmoid(out)[0][0].cpu().numpy())
            else:
                prob = float(model.predict(img_array, verbose=0)[0][0])
                
            label = 'FAKE' if prob > 0.5 else 'REAL'
            confidence = prob if prob > 0.5 else 1 - prob
            results.append({
                'model_id': model_id,
                'model_name': info['name'],
                'subtitle': info['subtitle'],
                'label': label,
                'confidence': round(confidence * 100, 2),
                'raw_score': round(prob, 4),
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'model_id': model_id,
                'model_name': info['name'],
                'subtitle': info['subtitle'],
                'status': 'error',
                'error': str(e)
            })

    valid = [r for r in results if r.get('status') == 'success']
    fake_votes = sum(1 for r in valid if r['label'] == 'FAKE')
    ensemble_label = 'FAKE' if fake_votes > len(valid) / 2 else 'REAL'
    ensemble_confidence = round(
        sum(r['confidence'] for r in valid) / len(valid) if valid else 0, 2
    )

    return jsonify({
        'type': 'image',
        'results': results,
        'ensemble': {
            'label': ensemble_label,
            'confidence': ensemble_confidence,
            'votes': {'fake': fake_votes, 'real': len(valid) - fake_votes}
        }
    })

# ─── API: Audio Predict ──────────────────────────────────────────
@app.route('/api/predict/audio', methods=['POST'])
def api_predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file.filename or not allowed_audio(file.filename):
        return jsonify({'error': 'Invalid file. Upload WAV, MP3, OGG, FLAC, or M4A.'}), 400

    audio_bytes = file.read()
    results = []

    for model_id, info in AUDIO_MODELS.items():
        try:
            features = extract_audio_features(audio_bytes, file.filename)
            model = get_model(model_id)
            rf_prob = model.predict_proba(features)[0]
            # Class 1 = FAKE, Class 0 = REAL (adjust if needed)
            fake_prob = float(rf_prob[1]) if len(rf_prob) > 1 else float(rf_prob[0])
            label = 'FAKE' if fake_prob > 0.5 else 'REAL'
            confidence = fake_prob if fake_prob > 0.5 else 1 - fake_prob
            results.append({
                'model_id': model_id,
                'model_name': info['name'],
                'subtitle': info['subtitle'],
                'label': label,
                'confidence': round(confidence * 100, 2),
                'raw_score': round(fake_prob, 4),
                'status': 'success'
            })
        except Exception as e:
            results.append({
                'model_id': model_id,
                'model_name': info['name'],
                'subtitle': info['subtitle'],
                'status': 'error',
                'error': str(e)
            })

    valid = [r for r in results if r.get('status') == 'success']
    if valid:
        fake_votes = sum(1 for r in valid if r['label'] == 'FAKE')
        ensemble_label = 'FAKE' if fake_votes > len(valid) / 2 else 'REAL'
        ensemble_confidence = round(
            sum(r['confidence'] for r in valid) / len(valid), 2
        )
    else:
        ensemble_label = 'UNKNOWN'
        ensemble_confidence = 0
        fake_votes = 0

    return jsonify({
        'type': 'audio',
        'results': results,
        'ensemble': {
            'label': ensemble_label,
            'confidence': ensemble_confidence,
            'votes': {'fake': fake_votes, 'real': len(valid) - fake_votes}
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
