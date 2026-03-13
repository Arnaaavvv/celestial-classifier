# 🌌 Celestial Object Classifier

A machine learning web application that classifies celestial objects from the Sloan Digital Sky Survey (SDSS) into three categories : **Galaxy**, **Star** or **Quasar** 

---

## 📌 Overview

| | |
|---|---|
| **Dataset** | SDSS Star Classification (100,000 objects) |
| **Model** | Logistic Regression |
| **Features** | 6 (u, g, r, i, z, redshift) |
| **Framework** | Streamlit |

---

## 🗂️ Project Structure

```
celestial-classifier/
├── app.py                      
├── celestial_object.ipynb      
├── star_model.pkl              
├── requirements.txt            
├── .gitignore
└── README.md
```

---

## 🔭 Features Used

| Feature | Description | Valid Range |
|---|---|---|
| `u` | Ultraviolet band magnitude | 14.0 – 35.0 |
| `g` | Green band magnitude | 14.0 – 35.0 |
| `r` | Red band magnitude | 14.0 – 35.0 |
| `i` | Near-infrared band magnitude | 14.0 – 35.0 |
| `z` | Infrared band magnitude | 14.0 – 35.0 |
| `redshift` | How fast the object moves away from us | −0.1 – 8.0 |


## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Arnaaavvv/celestial-classifier.git
cd celestial-classifier
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
plotly
```

---

## 🧠 Model Pipeline


### Prediction flow

```
Raw input → Clean negatives → Scale features → Predict → Decode label
```

---

## 🌠 Classes

| Class | Description | Typical Redshift |
|---|---|---|
| 🌌 **GALAXY** | Vast system of stars, gas, and dark matter | 0.01 – 2.0 |
| ⭐ **STAR** | Luminous sphere of plasma in the Milky Way | ≈ 0 (−0.01 to 0.01) |
| ✨ **QSO** | Extremely luminous active galactic nucleus powered by a supermassive black hole | 0.1 – 7.0+ |

---

## ⚠️ Known Limitations
 
- **Linear decision boundary** : Logistic Regression assumes a linear relationship between features and the target.
- **SDSS-specific model** : The model was trained exclusively on SDSS data. Photometric readings from other telescopes (e.g. Hubble, JWST) may have different calibrations.
- **No uncertainty quantification** : The confidence score shown is the raw `predict_proba` output from Logistic Regression, which tends to be overconfident. It should be treated as a relative indicator, not an absolute probability.
- **Sentinel value handling** : The app replaces negative band values with training-set medians. If an entire observation is corrupted, the prediction will be based mostly on redshift alone.
- **Class imbalance** : The dataset is imbalanced (GALAXY 59%, STAR 22%, QSO 19%). The model may be slightly biased toward predicting GALAXY for ambiguous objects.

---

## 📁 Dataset Credits

**Source:** [Sloan Digital Sky Survey : Star Classification Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)

- 100,000 observations
- 3 classes: GALAXY (59%), STAR (22%), QSO (19%)

---
If you liked this repository, kindly give it a star ⭐