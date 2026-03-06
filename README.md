# 🦟 Malaria Detection Project


A deep learning-powered web application that detects malaria from blood cell images using a fine-tuned **MobileNetV2** model. Upload a blood smear image and get an instant prediction — **Parasitized** or **Uninfected**.

---

## 📸 Demo

> Upload a blood cell image → Get instant malaria diagnosis with confidence score.

---

## 🗂️ Project Structure

```
malaria-detection-project/
│
├── MobileNetV2_final.h5       # Trained MobileNetV2 model weights
├── app.py                     # Flask backend server
├── index.html                 # Frontend UI
├── static.js                  # Frontend JavaScript logic
├── model training.ipynb       # Jupyter notebook for model training
└── README.md
```

---

## 🧠 How It Works

1. **Model Training** — A MobileNetV2 model is fine-tuned on the [NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria), which contains over 27,000 cell images split equally between parasitized and uninfected classes.
2. **Backend** — A Flask server loads the trained `.h5` model and exposes a REST API endpoint for image inference.
3. **Frontend** — A clean HTML/JS interface allows users to upload blood smear images and view predictions in real time.

---

## ⚙️ Tech Stack

| Component        | Technology               |
|-----------------|--------------------------|
| Model Architecture | MobileNetV2 (Transfer Learning) |
| Deep Learning Framework | TensorFlow / Keras |
| Backend | Flask (Python) |
| Frontend | HTML, JavaScript |
| Model File | `.h5` (HDF5 format) |
| Training Notebook | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/diksham-coder/malaria-detection-project.git
cd malaria-detection-project

# 2. Install dependencies
pip install flask tensorflow numpy pillow

# 3. Run the Flask app
python app.py
```

### Usage

1. Open your browser and go to `http://localhost:5000`
2. Upload a blood smear cell image (`.jpg`, `.png`)
3. Click **Predict** to get the result

---

## 🔬 Model Details

| Property         | Value                    |
|-----------------|--------------------------|
| Base Model       | MobileNetV2 (ImageNet weights) |
| Input Size       | 224 × 224 × 3            |
| Output Classes   | 2 (Parasitized / Uninfected) |
| Optimizer        | Adam                     |
| Loss Function    | Binary Crossentropy      |
| Saved Format     | `.h5`                    |

---

## 📊 Results

| Metric     | Score  |
|------------|--------|
| Accuracy   | ~95%+  |
| Precision  | High   |
| Recall     | High   |

> Refer to `model training.ipynb` for full training logs, accuracy/loss curves, and evaluation metrics.

---

## 📁 Dataset

This project uses the **NIH Malaria Cell Images Dataset**:
- 13,779 parasitized cell images
- 13,779 uninfected cell images
- [Download from Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

---

## 🙏 Acknowledgements

- [NIH – National Library of Medicine](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html) for the dataset
- [TensorFlow / Keras](https://www.tensorflow.org/) for the deep learning framework
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Howard et al., Google

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**diksham-coder**  
GitHub: [@diksham-coder](https://github.com/diksham-coder)

---

> ⚠️ **Disclaimer:** This tool is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis.
