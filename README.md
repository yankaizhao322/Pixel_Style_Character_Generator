# Pixel-style Character Generator based on DCGAN


Authors:
--------
- Yankai Zhao
- Wenqi Liu
- DSCI 498 - Lehigh University

Project Summary:
----------------
This project aims to automate the generation of pixel-style characters for games using a Conditional Deep Convolutional GAN (cDCGAN) model.

Traditionally, pixel art creation is manual and time-consuming. Our system enables users to generate diverse pixel characters (Monster, Human, Item, Equipment) by selecting a random seed and character type through a simple web interface.

Project Features:
-----------------
- Dataset sourced from Kaggle Pixel Art Characters (16x16 pixels, 89,000+ images).
- Generator conditioned on character types for controlled output.
- Deployed via Streamlit for user interaction and feedback collection.
- Feedback automatically saved to a Google Sheet via API.

Folder Structure:
-----------------
- backend/: Model training, generator code, and deployment utilities.
- frontend/: HTML, CSS, and JS files for web interface.
- dataset/: Dataset containing pixel character images.
- static/: Generated images storage.
- feedback/: Collected user feedback.

Software Requirements:
-----------------------
- Python 3.10+
- PyTorch
- torchvision
- Flask
- Streamlit
- gspread
- google-auth
- numpy
- PIL (pillow)

How to Run Locally:
-------------------
1. Clone this repository:

```bash
git clone https://github.com/yaz624/CSE498project/tree/master

streamlit run app.py --server.fileWatcherType none
```
or
```bash
python run.py
```
2.Install required Python packages:

```bash
pip install -r requirements.txt
```
(Or manually install: pip install torch torchvision flask streamlit gspread google-auth pillow numpy)

3.Prepare Dataset:

Download dataset (see data/readme_data.txt).

Place into dataset/pixel_art_data/.

4.Train or Load Model:

If training: run backend/train.py

If using pre-trained: ensure checkpoint in checkpoints/generator_epoch_460.pth

5.Run the Application:

For local Streamlit app:

```bash
streamlit run app.py
```

6.Deployment:

Deploy Streamlit app to Streamlit Community Cloud for public access.

Use `app.py` as the Streamlit entry point. The app expects:

- Python version: 3.10+
- Requirements file: `requirements.txt`
- Main file path: `app.py`

Feedback collection uses Streamlit secrets instead of a committed JSON credential file. In Streamlit
Community Cloud, add the contents shown in `.streamlit/secrets.toml.example` under App settings ->
Secrets. If secrets are not configured, character generation still works and only feedback submission
is disabled.

Important: never commit `backend/credentials/*.json` or `.streamlit/secrets.toml`. If a service
account key was ever committed publicly, rotate that key in Google Cloud before deploying.

# Contact:
For any issues, please contact:

Yankai Zhao: yaz624@lehigh.edu

# License:
This project is for educational and research purposes only.

![屏幕截图 2025-04-26 140442](https://github.com/user-attachments/assets/1d33d683-f050-4616-853f-7fab0cd02771)
