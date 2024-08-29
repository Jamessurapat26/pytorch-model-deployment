import streamlit as st
import requests
from PIL import Image
import io

# Define the URL of your Flask API
# try:
#     url = 'http://host.docker.internal:5000/predict'
# except:
url = 'http://localhost:5000/predict'

# Create the Streamlit interface
st.title("Image Prediction App")

uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Create a button to trigger the prediction
    if st.button('Predict'):
        # Prepare the file for the API request
        files = {'image': uploaded_file.getvalue()}

        # Make the API request
        try:
            response = requests.post(url, files=files)
            response.raise_for_status()  # Raise an exception for bad responses

            # Display the prediction results
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Confidence: {result['confidence']:.2f}")
            st.info(f"Class Index: {result['class_index']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the API: {e}")
        except KeyError:
            st.error("Unexpected response format from the API")