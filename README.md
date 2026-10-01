# Cacao Disease Detection

A FastAPI web app that detects cacao plant diseases from uploaded images using a MobileNetV3 model.

## Features

- Upload a cacao pod image from the browser
- Run disease classification through the trained model
- View prediction label, confidence, and probability breakdown
- Clean, responsive dashboard-style interface

## Project structure

```text
.
├── app/
│   ├── main.py
│   ├── model.py
│   └── static/
│       ├── index.html
│       └── styles.css
├── models/
│   └── cacao_disease_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

Python 3.10+ recommended.

Install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

From the project root:

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```text
http://localhost:8000
```

## API endpoint

The web interface calls the prediction API internally:

```http
POST /predict
```

Send an image file as multipart form data with the field name:

```text
file
```

Example using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/cacao-image.jpg"
```

## Model

The trained model is stored in:

```text
models/cacao_disease_model.pth
```

The app expects the model file to be present before startup.

## Notes

The current model supports the following classes:

- Black Pod Rot
- Healthy
- Pod Borer

If the model file is missing or the image is not valid, the app returns a clear error message.
