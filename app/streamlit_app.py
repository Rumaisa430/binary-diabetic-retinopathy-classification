"""

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array

# 1. Page Configuration
st.set_page_config(
    page_title='Diabetic Retinopathy Detection',
    page_icon='👁️',
    layout='centered',
)

# 2. Load the trained ResNet50 model (cached so it only loads once)
@st.cache_resource
def load_trained_model():
  model_path = 'models/resnet50_aptos.keras'
  return tf.keras.models.load_model(model_path)


# Load model safely
try:
  model = load_trained_model()
  model_loaded = True
except Exception as e:
  model_loaded = False
  model_error = e

# 3. UI Layout
st.title('👁️ Diabetic Retinopathy Detection')
st.write(
    'Upload a retinal scan image below to analyze it using your trained'
    ' **ResNet50** model.'
)

if not model_loaded:
  st.error(
      f'Could not load model from `models/resnet50_aptos.keras`. Error:'
      f' {model_error}'
  )

# File uploader widget
uploaded_file = st.file_uploader(
    'Choose a retina image...', type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
  # Display the uploaded image
  image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption='Uploaded Retina Scan', use_column_width=True)

  # Prediction Button
  if st.button('Run Prediction', type='primary'):
    if not model_loaded:
      st.error('Fix model loading error before running predictions.')
    else:
      with st.spinner('Analyzing retinal scan...'):
        # Preprocess the image to match training requirements
        img_resized = image.resize((224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        score = float(predictions[0][0])

        # Interpret output (assuming binary sigmoid model)
        if score > 0.5:
          diagnosis = 'Diabetic Retinopathy Detected'
          confidence = score * 100
          st.error(f'### ⚠️ Result: {diagnosis}')
        else:
          diagnosis = 'No Diabetic Retinopathy (Normal)'
          confidence = (1 - score) * 100
          st.success(f'### ✅ Result: {diagnosis}')

        # Show confidence metrics
        st.metric(label='Confidence', value=f'{confidence:.2f}%')
        with st.expander('See Technical Details'):
          st.write(f'Raw Model Output Score: `{score:.4f}`')

          """



import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input


# 1. Page Configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="centered",
)


# 2. Load trained ResNet50 model
@st.cache_resource
def load_trained_model():

    model_path = "models/resnet50_aptos.keras"

    return tf.keras.models.load_model(model_path)


# Load model
try:

    model = load_trained_model()
    model_loaded = True

except Exception as e:

    model_loaded = False
    model_error = str(e)


# 3. UI Layout
st.title("👁️ Diabetic Retinopathy Detection")

st.write(
    "Upload a retinal fundus image to detect "
    "Diabetic Retinopathy using ResNet50."
)


# Show model status
if not model_loaded:

    st.error(f"Model loading failed: {model_error}")


# 4. Image Upload
uploaded_file = st.file_uploader(
    "Choose a retinal image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        # Read uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(
            image,
            caption="Uploaded Retinal Image",
            use_container_width=True
        )

        # Prediction button
        if st.button("Run Prediction", type="primary"):

            if not model_loaded:

                st.error("Please fix the model loading error.")

            else:

                with st.spinner("Analyzing retinal image..."):

                    # Resize image
                    img_resized = image.resize((224, 224))

                    # Convert to NumPy array
                    img_array = np.array(
                        img_resized,
                        dtype=np.float32
                    )

                    # Add batch dimension
                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    # ResNet50 preprocessing
                    img_array = preprocess_input(img_array)

                    # Predict
                    predictions = model.predict(
                        img_array,
                        verbose=0
                    )

                    # Extract DR probability
                    score = float(predictions[0][0])

                    # Classification
                    if score >= 0.5:

                        diagnosis = "DR Detected"

                        confidence = score * 100

                        st.error(
                            "### ⚠️ Diabetic Retinopathy Detected"
                        )

                    else:

                        diagnosis = "No DR"

                        confidence = (1 - score) * 100

                        st.success(
                            "### ✅ No Diabetic Retinopathy Detected"
                        )

                    # Confidence
                    st.metric(
                        label="Prediction Confidence",
                        value=f"{confidence:.2f}%"
                    )

                    # Probability details
                    st.subheader("Prediction Details")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "DR Probability",
                        f"{score * 100:.2f}%"
                    )

                    col2.metric(
                        "No DR Probability",
                        f"{(1 - score) * 100:.2f}%"
                    )

                    # Technical details
                    with st.expander("Technical Details"):

                        st.write(f"Predicted Class: {diagnosis}")

                        st.write(f"Raw Model Output: {score:.4f}")

                        st.write("Model: ResNet50")

                        st.write("Image Size: 224 × 224")

                        st.write("Threshold: 0.50")

                    st.warning(
                        "This is an academic AI demonstration, "
                        "not a medical diagnosis. Consult an eye specialist."
                    )

    except (UnidentifiedImageError, OSError, ValueError):

        st.error("Invalid image. Please upload a valid JPG or PNG file.")
