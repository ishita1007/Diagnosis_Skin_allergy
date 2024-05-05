import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import base64  
import os  
import openai  # Import OpenAI API
# Set your OpenAI API key here
openai.api_key = "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"

# Set background image for the Streamlit page
background_image = r"C:\Users\ishit\OneDrive\Desktop\UI\skin_rashes_project_background_image.jpg"

# Mapping of class names to indices
CLASS_INDICES = {
    'Angiodema': 0,
    'Atopic Dermatitis': 1,
    'Cellulitis': 2,
    'Cercarical_Dermatitis': 3,
    'Contact Dermatitis': 4,
    'Shingles': 5,
    'Urticaria': 6
}

# Function to load and preprocess the image
def preprocess_image(image_file):
    # Save the uploaded image to a temporary file
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(image_file.read())
    
    # Load the saved image and preprocess it
    img = image.load_img(temp_image_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Remove the temporary image file
    os.remove(temp_image_path)
    
    return img_array

# Function to make prediction
def make_prediction(image_file, class_indices):
    # Load the model
    classifierLoad = tf.keras.models.load_model(r"C:\Users\ishit\OneDrive\Desktop\UI\Model_Classification.h5")
    
    # Preprocess the image
    test_image_array = preprocess_image(image_file)
    
    # Get the model prediction
    result = classifierLoad.predict(test_image_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(result)
    
    # Map the predicted class index to class name using class_indices
    predicted_class_name = [k for k, v in class_indices.items() if v == predicted_class_index][0]
    
    return predicted_class_name

# Define the Streamlit app
def main():
    st.set_page_config(page_icon="🧬",layout="wide")
    st.title("Skin Rashes Prediction")
    
    # Set background image
    st.markdown(
        f"""
        <style>
            .reportview-container {{
                background: url(data:image/jpeg;base64,{base64.b64encode(open(background_image, "rb").read()).decode()});
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction on the uploaded image
        if st.button("Make Prediction"):
            predicted_class_name = make_prediction(uploaded_file, CLASS_INDICES)
            st.write(f"Predicted Class: {predicted_class_name}")
    
    # ChatGPT integration
    with st.chat_message("user"):
        st.write("Hello! Ask me anything!")    
    user_input = st.text_area("Type your question or statement here:", "")
    if st.button("Show Results"):
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=user_input,
            max_tokens=50
        )
        st.text_area("Input:", user_input)
        st.text_area("Response:", response.choices[0].text.strip()) 
    
    # Clear button
    if st.button("Clear"):
        st.text_area("Type your question or statement here:", value="")
        st.text_area("Response:", value="")

if __name__ == "__main__":
    main()
