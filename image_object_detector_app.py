import cv2
import os
import streamlit as st
from ultralytics import YOLO
import uuid

def app():
    # App adalah kode yang akan selalu dijalankan oleh Python selama aplikasi berjalan
    st.header('Object Detection Web App for SIC Day 9') # Menuliskan header aplikasi
    st.subheader('Powered by YOLOv8')
    st.write('Welcome!')

    # Import model pakai package Ultralytics oleh YOLO
    model = YOLO('yolov8n.pt')
    object_names = list(model.names.values())

    print( "Name of the object predicted in YOLO are : "  )

    # Bikin form untuk input video
    with st.form("my_form"):

        # Ubah kode menjadi widget untuk input gambar
        uploaded_file = st.file_uploader("Upload image", type=['jpg', 'png'])
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=['person', 'car']) 
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        st.form_submit_button(label='Submit')
            
    if uploaded_file is not None: 
        # Generate unique filenames to avoid conflicts
        unique_id = str(uuid.uuid4().hex)[:8]  # Get first 8 characters of UUID for filename
        input_path = os.path.join(os.getcwd(), f"temp_{unique_id}.png")
        output_path = os.path.join(os.getcwd(), f"output_{unique_id}.png")

        # Ubah kode di bawah untuk prediksi object dalam image aja
        #------------------------------------------------------------
        try:
            with open(input_path, "wb") as temp_file:
                # Baca video yang di upload
                temp_file.write(uploaded_file.read())

            image_stream = cv2.imread( input_path )
            # Karena hasil darim cv2.imread itu gambar dalam BGR 
            # Kita ubah hasil gambar jadi RGB
            #image_stream = cv2.cvtColor( image_stream , cv2.COLOR_BGR2RGB )

            #video_stream = cv2.VideoCapture(input_path)
            #width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
            #height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            #fourcc = cv2.VideoWriter_fourcc(*'h264') 
            #fps = int(video_stream.get(cv2.CAP_PROP_FPS)) 

            #out_video = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height)) 

            with st.spinner('Processing object detection...'): 
                # Give notification to customer that video predicted now

                result = model(image_stream)

                frame= image_stream

                for detection in result[0].boxes.data:
                        # Making bounding box for location of object in videos
                        x0, y0 = (int(detection[0]), int(detection[1])) # Coordinate Top Left of bounding box
                        x1, y1 = (int(detection[2]), int(detection[3])) # Top Right of bounding box
                        score = round(float(detection[4]), 2)
                        cls = int(detection[5])
                        object_name =  model.names[cls]
                        label = f'{object_name} {score}'
                        
                        # Check if the model predicts object with confidence higher than min_confidence
                        if model.names[cls] in selected_objects and score > min_confidence:
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 5) # Bikin bounding box dengan sintaks : cv2.rectangle( gambar , (koordinat pojok kiri atas) , (koordinat pojok kanan bawah) , (warna bounding box) , (lebar garis bounding box))
                            cv2.putText(frame, label, (x0, y0 - 10), # Bikin text di atas bounding box
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        else:
                            continue

                # Setelah kasih bounding box ke gambar lalu export gambar ke output path


                cv2.imwrite( output_path , frame   )

            # Delete temporary files after processing
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                st.image(output_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()
