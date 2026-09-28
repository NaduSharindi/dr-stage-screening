# Diabetic Retinopathy Stage Screening (coursework prototype)
Multi-task transfer-learning CNN (MobileNetV2) trained on the APTOS 2019 Blindness Detection dataset (Kaggle).
Features: image-quality gate, CLAHE + Ben Graham preprocessing, test-time augmentation, Monte-Carlo-dropout uncertainty, Grad-CAM explanations and automatic triage.

Test results: accuracy 0.809 | macro-F1 0.659 | QWK 0.889

Run locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`

**Research/education prototype only - not a medical device.**
