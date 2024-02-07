from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit as st
import base64
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import requests
import time
from PIL import Image, ImageDraw
import io

def set_background():
    # GitHub raw content URL of the background image
    bg_image_url = "https://github.com/aliyassine1/streamlit_LiteMedSAM_quantized/blob/6c86d9a4a6e988dc6d832d6fab1ff6f3d80168a5/bg.png"

    # Use the URL in the style tag
    style = f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, 0.45), rgba(255, 255, 255, 0.45)), url('{bg_image_url}');
            background-size: cover;
        }}
        .textbox {{
            background-color: rgba(255, 255, 255, 0.9);  /* White background with 75% opacity */
            border-radius: 10px;  /* Rounded corners */
            padding: 10px;  /* Some padding around the text */
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


# Set API endpoints
api_endpoint_lite = "https://ay.us-east-1.modelbit.com/v1/apply_segmentation_and_blend/latest"
api_endpoint_quant = "https://ay.us-east-1.modelbit.com/v1/apply_segmentation_and_blend_fast/latest"

api_endpoint_multi_mask_lite="https://ay.us-east-1.modelbit.com/v1/apply_segmentation_and_blend_multiple_boxes/latest"
api_endpoint_multi_mask_quant="https://ay.us-east-1.modelbit.com/v1/apply_segmentation_and_blend_fast_multiple_boxes/latest"

st.set_page_config(layout='wide')
set_background()


css_text = """
<style>
.textbox {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
</style>
"""
css_radio = """
<style>
div.row-widget.stRadio > div { 
    background-color: rgba(255, 255, 255, 0.85); 
    border-radius: 10px; 
    padding: 10px; 
}
</style>
"""


st.title("Welcome to MedSam")
# Use a markdown with the custom class "textbox" for the text that you want to place on the opaque white box.
st.markdown('<div class="textbox">Upload an image, then click on two points to delineate the region of interest. For Multi-mask modes click twice to define a region, then click randomly one more time (1 dummy click) before defining the next region. Repeat this process for each region you wish to delineate. You can check the output to see your bounding boxes.</div>', unsafe_allow_html=True)

st.markdown(css_radio, unsafe_allow_html=True)

# Use markdown for the label with the custom class "textbox"
st.markdown('<div class="textbox">Please Choose the application mode. The APIs run on a CPU :</div>', unsafe_allow_html=True)
app_mode = st.radio("", ("MedSamLite", "MedSamLiteQuant (faster)","Multi-Mask on one image","Multi-Mask on one image (faster)"))



# Helper function to process image and make API request
def process_and_segment(image, bounding_box, api_endpoint):
    _, image_bytes = cv2.imencode('.png', np.asarray(image))
    image_bytes = image_bytes.tobytes()
    image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

    api_data = {"data": [image_bytes_encoded_base64, bounding_box]}
    response = requests.post(api_endpoint, json=api_data)
    response_data = response.json()

    return response_data

# Functionality for MedSamLite and MedSamLiteQuant (faster)
def medsam_segmentation(app_mode,file):

    if 'coordinates' not in st.session_state:
        st.session_state['coordinates'] = []

    if file is not None:
        image = Image.open(file).convert('RGB')
        value = streamlit_image_coordinates(image)

        if value is not None:
            if len(st.session_state['coordinates']) < 2:
                st.session_state['coordinates'].append(value)

            if len(st.session_state['coordinates']) == 2:
                x_coords = [coord['x'] for coord in st.session_state['coordinates']]
                y_coords = [coord['y'] for coord in st.session_state['coordinates']]
                bounding_box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                st.write(f"Coordinates (xmin, ymin, xmax, ymax): {bounding_box}")

                if st.button('Segment Region'):
                    start_time = time.time()
                    api_endpoint = api_endpoint_lite if app_mode == "MedSamLite" else api_endpoint_quant
                    response_data = process_and_segment(image, bounding_box, api_endpoint)

                    end_time = time.time()

                    blended_image_encoded = response_data['data'][2]
                    mask_image_encoded = response_data['data'][1]

                    blended_image_bytes = base64.b64decode(blended_image_encoded)
                    blended_image = cv2.imdecode(np.frombuffer(blended_image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    st.image(blended_image)

                    mask_image_bytes = base64.b64decode(mask_image_encoded)
                    mask_image = cv2.imdecode(np.frombuffer(mask_image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    _, mask_image_buffer = cv2.imencode('.png', mask_image)
                    st.download_button(label="Download Mask Generated", data=mask_image_buffer.tobytes(), file_name="mask_image.png", mime="image/png")

                if st.button('Reset Coordinates'):
                    st.session_state['coordinates'] = []

def multi_mask_segmentation(app_mode, file):
    if 'coordinates' not in st.session_state:
        st.session_state['coordinates'] = []

    if file is not None:
        # Display the uploaded image
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # Get user clicks on the image to define bounding boxes
        key = f"image_coordinates_{len(st.session_state['coordinates'])}"
        value = streamlit_image_coordinates(image, key=key)

        # Save the coordinate on user click
        if value is not None:
            st.session_state['coordinates'].append((value['x'], value['y']))

        # Display coordinates and calculate bounding boxes
        bounding_boxes = []
        if len(st.session_state['coordinates']) % 2 == 0 and st.session_state['coordinates']:
            for i in range(0, len(st.session_state['coordinates']), 2):
                x1, y1 = st.session_state['coordinates'][i]
                x2, y2 = st.session_state['coordinates'][i + 1]
                xmin = min(x1, x2)
                ymin = min(y1, y2)
                xmax = max(x1, x2)
                ymax = max(y1, y2)
                bounding_boxes.append((xmin, ymin, xmax, ymax))
                st.write(f"Bounding Box {i // 2 + 1}: ({xmin}, {ymin}, {xmax}, {ymax})")

        # The 'Segment' button triggers the API call and processes the image
        if st.button('Segment') and bounding_boxes:
            api_endpoint = api_endpoint_multi_mask_lite if app_mode == "Multi-Mask on one image" else api_endpoint_multi_mask_quant
            # Convert the image to base64 for API request
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_base64_encoding = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Prepare the API request data
            api_data = {"data": [image_base64_encoding, bounding_boxes]}
            response = requests.post(api_endpoint, json=api_data)
            response_data = response.json()

            # Process the response and display/download masks
            if 'data' in response_data:
                for i, mask_encoded in enumerate(response_data['data']):
                    mask_bytes = base64.b64decode(mask_encoded)
                    mask_image = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    st.image(mask_image, caption=f'Mask {i + 1}')

                    _, mask_image_buffer = cv2.imencode('.png', mask_image)
                    st.download_button(label=f"Download Mask {i + 1}",
                                       data=mask_image_buffer.tobytes(),
                                       file_name=f"mask_{i + 1}.png",
                                       mime="image/png")

            # Clear coordinates after segmentation
            st.session_state['coordinates'] = []

        # A button to clear the coordinates and reset the process
        if st.button('Reset Coordinates'):
            st.session_state['coordinates'] = []


def multi_mask_segmentation(app_mode, file):
    if 'bounding_boxes' not in st.session_state:
        st.session_state['bounding_boxes'] = []

    if file is not None:
        # Display the uploaded image
        image = Image.open(file).convert('RGB')


        # Allow user to click on the image to define bounding boxes
        # Use the length of the bounding_boxes list to generate a unique key
        key = f"image_coordinates_{len(st.session_state['bounding_boxes']) // 2}"
        print(key)
        value = streamlit_image_coordinates(image, key=key)

        # If user clicks on the image, save the coordinate
        if value is not None:
            st.session_state['bounding_boxes'].append((value['x'], value['y']))


        # Display the coordinates as bounding boxes
        if len(st.session_state['bounding_boxes']) > 0:
            for i in range(0, len(st.session_state['bounding_boxes']), 2):
                st.write(f"Bounding Box {i // 2 + 1}: {st.session_state['bounding_boxes'][i:i + 2]}")

        # The Segment button triggers the API call and processes the image
        if st.button('Segment'):
            api_endpoint = api_endpoint_multi_mask_lite if app_mode == "Multi-Mask on one image" else api_endpoint_multi_mask_quant
            if len(st.session_state['bounding_boxes']) % 2 == 0:

                # Initialize an empty list to hold the formatted bounding boxes
                formatted_bounding_boxes = []

                # Iterate over the bounding_boxes in pairs
                for i in range(0, len(st.session_state['bounding_boxes']), 2):
                    x1, y1 = st.session_state['bounding_boxes'][i]
                    x2, y2 = st.session_state['bounding_boxes'][i + 1]

                    # Determine xmin, ymin, xmax, ymax
                    xmin = min(x1, x2)
                    ymin = min(y1, y2)
                    xmax = max(x1, x2)
                    ymax = max(y1, y2)

                    # Append the formatted bounding box to the list
                    formatted_bounding_boxes.append((xmin, ymin, xmax, ymax))

                # Now we have bounding boxes in the desired format
                bounding_boxes = formatted_bounding_boxes


                # Convert the image to base64 for API request
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                image_base64_encoding = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Prepare the API request data
                api_data = {"data": [image_base64_encoding, bounding_boxes]}
                response = requests.post(api_endpoint, json=api_data)
                response_data = response.json()

                # Process the response and display/download masks
                if 'data' in response_data:
                    for i, mask_encoded in enumerate(response_data['data']):
                        mask_bytes = base64.b64decode(mask_encoded)
                        mask_image = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                        st.image(mask_image, caption=f'Mask {i + 1}')

                        # Provide a button to download each mask image
                        _, mask_image_buffer = cv2.imencode('.png', mask_image)
                        st.download_button(label=f"Download Mask {i + 1}",
                                           data=mask_image_buffer.tobytes(),
                                           file_name=f"mask_{i + 1}.png",
                                           mime="image/png")

            # Clear the bounding boxes after segmentation

            st.session_state['bounding_boxes'] = []

        # A button to clear the coordinates and reset the process
        if st.button('Reset Bounding Boxes'):
            st.session_state['bounding_boxes'] = []


# Get the file from the uploader
file = st.file_uploader('Upload an image', type=['jpeg', 'jpg', 'png'])

# Call the segmentation function and pass in the uploaded file
if app_mode == "MedSamLite":
    medsam_segmentation(app_mode, file)

elif app_mode == "MedSamLiteQuant (faster)":
    medsam_segmentation(app_mode, file)

elif app_mode == "Multi-Mask on one image":
    multi_mask_segmentation(app_mode, file)

elif app_mode == "Multi-Mask on one image (faster)":
    multi_mask_segmentation(app_mode, file)




