import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('digit.hdf5')

st.title('Digit Recognizer:')

# Create a canvas component
canvas_result = st_canvas(stroke_width=10, stroke_color='#ffffff',
                          background_color='#000000',
                          height=200, width=200,
                          drawing_mode='freedraw')

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    img = cv2.resize(canvas_result.image_data.astype(np.uint8), (28, 28))
    img_rescaling = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_NEAREST)

    if st.button("Predict"):
        x_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y_pred = model.predict(x_img.reshape(1, 28, 28))
        y_pred = np.argmax(y_pred, axis=1)
        st.markdown("<h1 style='text-align: center; color: red;'>********Result:*********</h1>", unsafe_allow_html=True)
        st.header("The Predicted Value is:")
        st.title(y_pred[0])
