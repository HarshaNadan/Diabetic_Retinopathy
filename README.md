**Deep Learning–Based Diabetic Retinopathy Detection**

Early detection of Diabetic Retinopathy (DR) using deep learning on retinal fundus images. This repo fine-tunes EfficientNetB0 and DenseNet121, evaluates multi-class performance (DR grades 0–4), and explains predictions using Grad-CAM (with optional SHAP planned).

**Highlights**
- Dataset: APTOS 2019 Blindness Detection (3,662 images, labels 0–4)
- Models: EfficientNetB0, DenseNet121 (transfer learning)
- Methods: class weighting, augmentation, early stopping, Grad-CAM
- Validation Accuracy: ~74%
- Known gap: poor recall on Severe (3) and Proliferative (4) due to class imbalance

**Project Goals**
1. Build a reproducible pipeline to classify DR severity (0–4) from fundus photos.
2. Compare EfficientNetB0 and DenseNet121 with transfer learning.
3. Address class imbalance with class weights and augmentation.
4. Improve clinical trust via Grad-CAM visualizations (and SHAP in future).

**Repo Structure**
- Diabetic_Retinopathy_Detection (1).ipynb : Main Jupyter notebook
- Diabetic_Retinopathy_Detection.docx : Project report (write-up)
- README.md : This file
- requirements.txt : (Optional) dependencies
- data/ : contains train.csv and train/ images

**Setup**
1) Create environment:
python -m venv .venv
.venv\Scripts\activate (Windows)
source .venv/bin/activate (macOS/Linux)
pip install --upgrade pip

2) Install dependencies:
pip install tensorflow keras numpy pandas scikit-learn opencv-python matplotlib pillow tqdm
Optional: pip install seaborn shap kaggle
Or: pip install -r requirements.txt

3) Launch Jupyter:
jupyter notebook
Open Diabetic_Retinopathy_Detection (1).ipynb and run cells top-to-bottom.

**Data**
Dataset: APTOS 2019 Blindness Detection. Labels:
0 – No DR, 1 – Mild, 2 – Moderate, 3 – Severe, 4 – Proliferative DR
Place train.csv and the train/ folder under ./data/ (or update notebook paths).

**Training & Evaluation**
- Image size: 224×224, RGB normalized [0,1]
- Augmentation: rotation, flips, zoom, brightness/contrast
- Split: stratified 80/20 train/val
- Optimizer: Adam, LR=1e-4
- Epochs: up to 15 with EarlyStopping
- Loss: categorical cross-entropy (consider Focal Loss)
- Imbalance: class weights + targeted augmentation
- Metrics: Accuracy, Precision, Recall, F1 (macro/weighted), confusion matrix
Steps: run preprocessing, train EfficientNetB0, train DenseNet121, evaluate + plots

**Explainability**
- Grad-CAM: implemented (heatmaps show regions of importance)
- SHAP: planned (pixel-level attribution)

**Results (Summary)**
- Validation accuracy: ~74% for both models
- Macro F1: ~0.40–0.42 (shows imbalance)
- Strengths: Class 0 (No DR), Class 2 (Moderate)
- Weaknesses: Class 3 (Severe), Class 4 (Proliferative)
- Grad-CAM: EfficientNetB0 → central focus, missed periphery; DenseNet121 → highlighted more clinically relevant features

**Limitations & Future Work**
- Class imbalance is the main limitation
- Planned improvements: Focal Loss, oversampling (SMOTE/GAN), ensembles (EfficientNet, DenseNet, ResNet, Inception), calibration (ECE, Brier score, temperature scaling), cross-dataset validation (EyePACS, Messidor), combine SHAP + Grad-CAM, optimize for edge deployment (quantization/distillation)

**Citations**
- Dataset: APTOS 2019 Blindness Detection (Kaggle)
- Models: EfficientNet, DenseNet
- See dissertation for full references

**License**
MIT (or your choice). Respect dataset terms.

**Contact**
Author: Sri Sai Harsha Nandan Vegireddi
For questions or collaboration, please open an issue or reach out directly.

