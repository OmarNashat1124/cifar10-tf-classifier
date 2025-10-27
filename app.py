import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🧠")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cifar10_model.h5') 
    return model

model = load_model()


class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


st.title("🚀 CIFAR-10 Image Classifier (CNN Model)")
st.write("Upload a 32×32 image — the model will predict which class it belongs to!")

uploaded_file = st.file_uploader(" Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption=' Uploaded Image', use_column_width=True)
    st.write("Processing...")

    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) 


    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100


    st.markdown("---")
    st.markdown(f"###  **Prediction:** `{pred_class.capitalize()}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")


    st.markdown("#### Class probabilities:")
    for i, c in enumerate(class_names):
        st.write(f"{c:12s}: {prediction[0][i]*100:.2f}%")

else:
    st.info("Upload an image to get started!")
