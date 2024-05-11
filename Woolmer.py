import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Function to load TensorFlow Lite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess input data
def preprocess_input(data):
    # Perform any necessary preprocessing (e.g., scaling, normalization)
    expected_shape = (1, 5)  # Example: Assuming input tensor shape is (1, 5)
    data = np.array(data, dtype=np.float32)
    data = np.array(data).reshape(expected_shape)
    # Return preprocessed data as a NumPy array
    return np.array(data)

# Function to run inference
def run_inference(interpreter, input_data):
    # Set input tensor
    input_details = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_details['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.get_tensor(output_details['index'])

    return output_data

# Streamlit UI
def main():
    page_bg_img = '''
    <style>
    body {
    background-image: url("Welcome.png");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    image = Image.open("logo-10.png")  # Replace "your_image_file_path.jpg" with the path to your image file
    image2 = Image.open("image-2.png")
    image3 = Image.open("injury.jpg")
    image4 = Image.open("fit.jpg") 
    # Define the layout
    col1, col2, col3 = st.columns([1, 25 ,1])  # Divide the page into two columns, with 1/5 and 4/5 widths
    col11, col12, col13, col14, col15 = col2.columns([1,1,2,1,1])
    col41, col42, col43 = col2.columns([1,30,1])
    _ ,col21, _ = st.columns([1,25,1])
    # Load TensorFlow Lite model
    interpreter = load_model()
    st.image(image, use_column_width= True)
    # User input
    #input_data = st.text_input("Input Data", "25, 23.5, 1, 8, 3")
    input_age = st.text_input("Player Age")
    input_height = st.text_input("Player Height (cm)")
    input_weight = st.text_input("Player Weight (kgs)")
    input_previous_injury = st.selectbox(
    'Player Recently Injured',
    ('Yes', 'No'))
    if input_previous_injury == "Yes":
        input_injury = 1
    else:
        input_injury = 0
    input_training_intensity = st.selectbox(
    'Training Intensity of Player (0 to 10)', ('0','1','2','3','4','5','6','7','8','9','10') )

    input_intensity = int(input_training_intensity)
    input_intensity = float(input_intensity/10)
    input_recover_days = st.selectbox(
    'Average Number of Days to Recover (Small Injuries)', ('1','2','3','4','5','6') )
    input_recovery = int(input_recover_days)
    
    # Run inference when button is clicked
    if st.button("Run Model"):
        # Run inference
        input_age_float = float(input_age)
        input_weight_float = float(input_weight)
        input_height_float = float(input_height)
        input_BMI = float(input_weight_float / (input_height_float)/100) ** 2 
        input_data = [input_age_float,input_BMI,input_injury,input_intensity,input_recovery]
        input_data = preprocess_input(input_data)
        output_data = run_inference(interpreter, input_data)
        if output_data < 0.5:
            st.write("The model indicates that you are fit to play without significant risk of injury")
            st.image(image4, use_column_width= True)
        else:
            st.write("The model predicts that you are at risk of injury")
            st.image(image3, use_column_width= True)
        #st.write("Output:", output_data)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.image(image2, use_column_width= True)

if __name__ == "__main__":
    main()
