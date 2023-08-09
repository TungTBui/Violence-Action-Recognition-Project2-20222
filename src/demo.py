import streamlit as st
import cv2
from PIL import Image 
import numpy as np
import keras
from collections import deque
import tempfile
import pickle
import time
import subprocess
import os


output_file = r"C:\\Users\\Admin\\Downloads\\Test_vid\\vid.mp4"
args = {
    "size": 128
}

st.title("Violent Action Recognition")
file_name = st.file_uploader("Upload file")
print(file_name)
if file_name == 'camera':
	file_name = 0
        
stframe = st.empty()
    
print("LOADING MODEL")
xgb_path = r".\\xgb.pkl"
model_xgb = pickle.load(open(xgb_path, "rb"))
model_inceptionv3 = keras.models.load_model(r"C:\\Users\\Admin\\Downloads\\inceptionv5.keras") 

print("LOADING DONE")

mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

tfile = tempfile.NamedTemporaryFile(delete=False) 
tfile.write(file_name.read())

video_file = open(tfile.name, 'rb')
video_bytes = video_file.read()

st.video(video_bytes)

vs = cv2.VideoCapture(tfile.name)

writer = None
(W, H) = (None, None)

with st.spinner('Analysing the video...'):

    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        
        if not grabbed:
            print("DONE")
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean

        # make predictions on the frame and then update the predictions
        
        preds = model_inceptionv3.predict(np.expand_dims(frame, axis=0))[0]
        i = (preds > 0.70)[1]
        label = i

        if label: # Violence prob
            text_color = (0, 0, 255) # red
            
        else:
            text_color = (0, 255, 0)

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX 

        cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3) 

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            writer = cv2.VideoWriter(output_file, fourcc, 30,(W, H), True)

        # write the output frame to disk
        writer.write(output)

        print("Shown")


# release the file pointersq
print("[INFO] cleaning up...")
writer.release()
vs.release()

st.success('Done!')

print("DONE")
st.subheader('Result:')

with open(output_file, 'rb') as v:
    st.video(v)


