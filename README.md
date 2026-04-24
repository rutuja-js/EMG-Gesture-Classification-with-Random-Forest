# EMG Gesture Classification with Random Forest

A machine learning project that classifies hand gestures (**Rest** vs **Fist**) from raw **EMG (Electromyography)** signal data using a **Random Forest** classifier. Built and run in Google Colab.

---

## 📌 Overview

Electromyography (EMG) measures the electrical activity produced by muscles. This project reads multi-channel EMG voltage signals and trains a supervised ML model to distinguish between two gestures:

| Label | Gesture |
|-------|---------|
| `1`   | Rest    |
| `2`   | Fist    |

---

## 📁 Repository Structure

```
bioinfo.rs/
├── ml_rf_rs.py       # Main ML pipeline script
└── README.md
```

> **Note:** The dataset (`AI MODEL - Sheet1.csv`) is uploaded at runtime via Google Colab's file uploader.

---

## 🔧 Tech Stack

- **Python 3**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — signal visualization
- **scikit-learn** — ML model (Random Forest), train/test split, evaluation
- **joblib** — model serialization

---

## 🚀 Getting Started

### 1. Open in Google Colab

This script is designed to run in [Google Colab](https://colab.research.google.com/). Open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1e_KbSqTJrUHc-Ms11bcJYHCon8366XkY)

### 2. Upload Your Dataset

When prompted, upload your CSV file. The expected format:

```
ch1_voltage, ch2_voltage, ch3_voltage, label
0.42, 0.31, 0.27, 1
0.48, 0.33, 0.29, 2
...
```

### 3. Run the Script

Execute `ml_rf_rs.py` cell by cell. The script will:
1. Load and inspect the dataset
2. Visualize raw EMG signals across 3 channels
3. Train a Random Forest model
4. Evaluate model performance
5. Predict gestures from new EMG readings
6. Save and reload the trained model

---

## 📊 Model Pipeline

```
Raw EMG CSV
    │
    ▼
Data Cleaning (dropna)
    │
    ▼
Feature Extraction: [ch1_voltage, ch2_voltage, ch3_voltage]
    │
    ▼
Train/Test Split (80/20)
    │
    ▼
Random Forest Classifier (100 estimators)
    │
    ▼
Evaluation: Accuracy, Confusion Matrix, Classification Report
    │
    ▼
Prediction → "Rest" or "Fist"
    │
    ▼
Model saved as emg_gesture_model.pkl
```

---

## 📈 Evaluation Metrics

The model outputs:
- **Accuracy Score**
- **Confusion Matrix**
- **Full Classification Report** (precision, recall, F1-score)

---

## 🔮 Inference Example

```python
new_data = [[0.42, 0.31, 0.27]]
gesture = model.predict(new_data)
print("Predicted Gesture:", "Fist" if gesture[0] == 2 else "Rest")
```

---

## 💾 Model Persistence

The trained model is saved and can be reloaded for future inference:

```python
import joblib

# Save
joblib.dump(model, 'emg_gesture_model.pkl')

# Load
model = joblib.load('emg_gesture_model.pkl')
```

---

## 🛠️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install locally (if not using Colab):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or file an issue.

---

## 📄 License

This project is open source. Add a license of your choice (e.g., MIT).

---

## 👩‍💻 Author

**rutuja-js** — [GitHub Profile](https://github.com/rutuja-js)
