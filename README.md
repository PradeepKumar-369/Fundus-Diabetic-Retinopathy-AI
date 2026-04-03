# 👁️ Diabetic Retinopathy Diagnostic AI (GWO-Optimized)

An end-to-end Deep Learning system that diagnoses Diabetic Retinopathy from retinal fundus images. This project features a custom EfficientNet model tuned using the **Grey Wolf Optimizer (GWO)** algorithm and includes a **Django** web dashboard for real-time inference and **Explainable AI (Grad-CAM)** visualizations.

## 🚀 Key Features
* **Deep Learning Architecture:** EfficientNet-B0 trained on a heavily augmented dataset across 5 clinical stages of the disease.
* **Swarm Intelligence Optimization:** Utilized a custom Grey Wolf Optimizer (GWO) script to dynamically find the mathematical optimum for Learning Rate and Dropout parameters, preventing memory overload and maximizing accuracy.
* **Explainable AI (XAI):** Integrated Grad-CAM to generate visual heatmaps. This allows medical professionals to see exactly which retinal lesions, exudates, or camera artifacts the AI focused on to make its diagnosis.
* **Web Dashboard:** A lightweight Django interface where users can upload raw images and instantly receive the diagnosis label, AI confidence score, and the visual heatmap.

## 🛠️ Tech Stack
* **AI/ML Core:** Python, TensorFlow, Keras, OpenCV, NumPy
* **Backend Web Server:** Django
* **Optimization:** Custom GWO Algorithm

## 📊 Model Performance
* Achieved **71.08% Validation Accuracy** across 5 complex clinical classes (Healthy, Mild, Moderate, Severe, Proliferative).
* GWO Optimized Parameters: Learning Rate `0.00080`, Dropout Rate `0.31`.
* System successfully balances diagnostic confidence with clinical caution, effectively identifying severe lesions while flagging visual artifacts via Grad-CAM.

## 💻 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/PradeepKumar-369/Fundus-Diabetic-Retinopathy-AI.git](https://github.com/PradeepKumar-369/Fundus-Diabetic-Retinopathy-AI.git)