import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load the model
model = load_model('digit.hdf5')

st.title('Digit Recognizer')

# Create a layout with two columns
col1, col2 = st.beta_columns([1, 1])

# Create a canvas component in the first column
with col1:
    st.header("Draw a digit:")
    canvas_result = st_canvas(stroke_width=10, stroke_color='#ffffff',
                              background_color='#000000',
                              height=200, width=200,
                              drawing_mode='freedraw')

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    img = cv2.resize(canvas_result.image_data.astype(np.uint8), (28, 28))
    img_rescaling = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)

# Create a button to predict in the second column
with col2:
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            x_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            y_pred = model.predict(x_img.reshape(1, 28, 28))
            y_pred = np.argmax(y_pred, axis=1)
            st.markdown("<h1 style='text-align: center; color: red;'>********Result:*********</h1>", unsafe_allow_html=True)
            st.header("The Predicted Value is:")
            st.title(y_pred[0])
        else:
            st.warning("Please draw a digit before predicting.")
