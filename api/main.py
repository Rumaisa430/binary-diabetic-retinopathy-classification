"""

FastAPI backend for diabetic retinopathy detection

from pathlib import Path

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException

from tensorflow.keras.applications.resnet50 import preprocess_input


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "resnet50_aptos.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["No DR", "DR"]


app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="Binary DR classification using ResNet50",
    version="1.0"
)


# Load model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Train the model first."
    )

model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/")
def home():

    return {
        "message": "Diabetic Retinopathy Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    try:

        image = Image.open(file.file).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image_array = np.array(image, dtype=np.float32)

        image_array = np.expand_dims(image_array, axis=0)

        # Same preprocessing as training
        image_array = preprocess_input(image_array)

        prediction = model.predict(
            image_array,
            verbose=0
        )

        dr_probability = float(prediction[0][0])

        if dr_probability >= 0.5:

            predicted_class = "DR"
            confidence = dr_probability

        else:

            predicted_class = "No DR"
            confidence = 1 - dr_probability

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "dr_probability": round(dr_probability * 100, 2),
            "no_dr_probability": round(
                (1 - dr_probability) * 100, 2
            )
        }

    except (UnidentifiedImageError, OSError):

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )
"""



"""FastAPI backend for diabetic retinopathy detection."""

from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException

from tensorflow.keras.applications.resnet50 import preprocess_input


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "resnet50_aptos.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["No DR", "DR"]


# Create FastAPI application
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="Binary DR Classification using ResNet50",
    version="1.0"
)


# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("ResNet50 model loaded successfully!")


# Home route
@app.get("/")
def home():

    return {
        "message": "Diabetic Retinopathy Detection API is running"
    }


# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check image extension
    allowed_extensions = [".jpg", ".jpeg", ".png"]

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG or PNG image."
        )

    try:

        # Read uploaded image
        contents = await file.read()

        # Open image
        image = Image.open(BytesIO(contents))

        # Convert image to RGB
        image = image.convert("RGB")

        # Resize image
        image = image.resize(IMAGE_SIZE)

        # Convert image to NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ResNet50 preprocessing
        image_array = preprocess_input(image_array)

        # Model prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        # Extract DR probability
        dr_probability = float(prediction[0][0])

        # Determine predicted class
        if dr_probability >= 0.5:

            predicted_class = CLASS_NAMES[1]
            confidence = dr_probability

        else:

            predicted_class = CLASS_NAMES[0]
            confidence = 1 - dr_probability

        # Return prediction result
        return {

            "predicted_class": predicted_class,

            "confidence": round(
                confidence * 100, 2
            ),

            "dr_probability": round(
                dr_probability * 100, 2
            ),

            "no_dr_probability": round(
                (1 - dr_probability) * 100, 2
            )

        }

    except (UnidentifiedImageError, OSError, ValueError):

        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid retinal image."
        )

    finally:
        await file.close()
