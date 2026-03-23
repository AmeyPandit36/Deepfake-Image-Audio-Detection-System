# DeepGuard — Build Complete ✅

## What Was Built

A full-stack **Flask + HTML/CSS/JS** web application for deepfake image detection, running at `http://localhost:5000`.

## Files Created

| File | Purpose |
|---|---|
| [app.py](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/app.py) | Flask server with routing + `/api/predict` endpoint |
| [requirements_app.txt](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/requirements_app.txt) | Flask + Pillow dependencies |
| [templates/base.html](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/templates/base.html) | Base layout (navbar, footer, scripts) |
| [templates/index.html](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/templates/index.html) | Landing page with hero, features, CTA |
| [templates/models.html](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/templates/models.html) | Model cards + Chart.js bar chart |
| [templates/predict.html](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/templates/predict.html) | Drag-and-drop upload + results panel |
| [static/css/style.css](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/static/css/style.css) | 500+ line dark glassmorphism CSS |
| [static/js/predict.js](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/static/js/predict.js) | Upload flow, API calls, results rendering |

## Phase 2: PyTorch Model Integration & Audio Support
To support diverse deepfake detection strategies, the app was heavily refactored:
1. **Multi-Modal Tabs:** The user interface now properly segments Image and Audio upload flows with dedicated tabs.
2. **PyTorch Integration:** A newly added **ResNet-50** PyTorch model was seamlessly integrated into the backend ensemble, complementing the existing Keras CNNs.
3. **MFCC Audio Preprocessing:** The backend endpoints were split (`/api/predict/image` and `/api/predict/audio`), using Librosa feature extraction to support the Scikit-Learn Random Forest audio classifier.

### Backend Enhancements ([app.py](file:///c:/Users/Amey%20Ravindra%20Pandit/OneDrive/Desktop/2b%20Projet/deepfake-image-detection/app.py))
- Split prediction logic appropriately based on modal type.
- Integrated `torch` and `torchvision`.
- Reconstructed the PyTorch `state_dict` loading sequence to match the modified ResNet-50 `Linear` head structure.

### User Interface Iteration
![Multi-Modal Predict Page](file:///C:/Users/Amey%20Ravindra%20Pandit/.gemini/antigravity/brain/db6c945d-aa86-46ed-85fc-38ce503437b3/predict_page_verified_1774279261180.png)
*The new multi-modal file upload interface with distinct tabs and drop zones.*

![Updated Models Directory](file:///C:/Users/Amey%20Ravindra%20Pandit/.gemini/antigravity/brain/db6c945d-aa86-46ed-85fc-38ce503437b3/models_page_verified_1774279243578.png)
*ResNet-50 displayed prominently along with updated dynamic `Chart.js` metrics.*

---

## Phase 3: Landing Page Polish & Responsive Fixes
After reviewing the initial landing page design, a critical structural bug was found where CSS grid properties were incorrectly applied directly to the inner text wrapper rather than the parent layout. This caused the hero text and visual cards to disconnect.
- **Structural Fix**: Re-packaged the hero section with a dedicated `<div class="hero-container">` to enforce the dual-column layout strictly between the textual content and the scan card visual.
- **Copy Enhancements**: The text strictly highlights the **Multi-Modal Image + Audio** aspect across all feature descriptions and the statistic bar. 
- **Mobile Support**: Added standard `@media` viewport queries to gracefully collapse the layout for standard 600px limits.

---

## Pages

- **`/`** — Hero section with scan card animation, stats bar, feature cards, how-it-works steps
- **`/models`** — Performance comparison chart (Chart.js), metrics table, 3 model detail cards with architecture flows and classification reports
- **`/predict`** — Drag-and-drop upload, image preview, model checkboxes, animated loading steps, ensemble verdict + per-model results

## Predict Page Screenshot

![Predict Page](file:///C:/Users/Amey Ravindra Pandit/.gemini/antigravity/brain/db6c945d-aa86-46ed-85fc-38ce503437b3/predict_page_1774275374047.png)

## Running the Application

```powershell
# From the project directory:
.venv39\Scripts\python.exe app.py
```
Then open: **http://localhost:5000**

## Model Loading Note

> Models are **lazy-loaded** on first prediction request to avoid slow startup. The first prediction will take longer as the `.h5` and `.pkl` files are loaded into memory. Subsequent predictions will be fast.

## How the Random Forest Works

The RF was trained on CNN feature vectors. On predict:
1. Custom CNN extracts a **512-dim feature vector** from the image
2. Feature is passed to the Random Forest for classification
3. RF returns `predict_proba` → REAL/FAKE label + confidence
