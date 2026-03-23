# Flask Deepfake Detection App

A full-stack Flask web application for deepfake image detection, featuring a premium dark-mode UI with navigation, model performance dashboards, and real-time prediction from all 3 trained models.

## Proposed Changes

### App Structure (new directory `app/` inside project root)

```
deepfake-image-detection/
├── app.py                     [NEW] Flask entry point
├── requirements_app.txt       [NEW] Flask + dependencies
├── templates/
│   ├── base.html              [NEW] Base layout, navbar (Home / Models / Predict)
│   ├── index.html             [NEW] Landing page with hero + feature cards
│   ├── models.html            [NEW] Model cards + performance metrics charts
│   └── predict.html           [NEW] Upload image → results from all models
└── static/
    ├── css/style.css          [NEW] Dark-theme, glassmorphism, responsive
    └── js/
        ├── main.js            [NEW] Upload flow, API call, results rendering
        └── charts.js          [NEW] Chart.js performance visualizations
```

---

### Backend

#### [NEW] app.py
- Flask app with 3 routes: `/` (home), `/models`, `/predict`
- REST API endpoints:
  - `GET /api/models` → returns performance metadata for all models
  - `POST /api/predict` → accepts uploaded image, runs all 3 models, returns JSON results
- **Model loading at startup** (lazy-loaded once on first request to avoid slow startup)
- Image preprocessing: PIL resize to 224×224, normalize to [0,1]
- For CNN models: use `model.predict()` → sigmoid score → REAL / FAKE
- For Random Forest: extract features from Custom CNN intermediate layer → feed to RF

> [!IMPORTANT]
> The `random_forest_model.pkl` is an ML model trained on CNN-extracted features (512-dim vectors). The app will use the Custom CNN to extract a feature vector, then pass it to the RF classifier.

---

### Frontend

#### templates/base.html
- Navbar with logo + links: Home, Models, Predict
- Google Fonts (Inter), dark background `#0a0a0f`
- Mobile hamburger menu

#### templates/index.html
- Hero section: headline, subheadline, CTA button → Predict page
- Stats bar: 3 models, 100K+ training images, 98.5% AUC
- Feature cards: CNN Architecture, Augmentation, Ensemble

#### templates/models.html
- 3 model cards with performance metrics
- Bar chart (Chart.js) comparing Accuracy, ROC AUC, AP Score across models
- Architecture details per model

#### templates/predict.html
- Drag-and-drop image upload zone
- Image preview before submission
- Submit button → loading spinner
- Results section: card per model showing REAL/FAKE + confidence bar

#### static/css/style.css
- CSS variables for dark theme
- Glassmorphism cards
- Animated gradient hero
- Responsive grid layouts

---

## Performance Metrics (hard-coded from notebook outputs)

| Model | Accuracy | ROC AUC | AP Score |
|---|---|---|---|
| Custom CNN (No Augmentation) | 86% | 0.985 | 0.984 |
| Custom CNN (With Augmentation) | 73% | 0.937 | 0.933 |
| Random Forest (on CNN features) | ~82%* | ~0.91* | ~0.91* |

*RF metrics estimated — will be fetched from pkl metadata or kept as approximate.

---

## Verification Plan

### Automated Tests
None existing in repo. No automated tests will be added — app is verified manually.

### Manual Verification (Browser)
1. Run the server: `python app.py` from the project root
2. Open browser → `http://localhost:5000`
3. **Home page**: verify hero, feature cards, and CTA button render
4. **Navigate to Models page** (`/models`): verify 3 model cards and Chart.js bar chart appear
5. **Navigate to Predict page** (`/predict`): upload a `.jpg` image → verify loading spinner shows → results appear with 3 model predictions
6. **Responsive test**: resize browser to mobile width and verify layout adapts
