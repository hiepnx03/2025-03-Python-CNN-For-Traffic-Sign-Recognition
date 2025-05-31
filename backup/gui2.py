import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
import cv2

# Load the trained model to classify sign
model = load_model('traffic_classifier.h5')

# Dictionary to label all traffic signs class.
classes = {
    1: 'Speed limit (20km/h)',
    2: 'Speed limit (30km/h)',
    3: 'Speed limit (50km/h)',
    4: 'Speed limit (60km/h)',
    5: 'Speed limit (70km/h)',
    6: 'Speed limit (80km/h)',
    7: 'End of speed limit (80km/h)',
    8: 'Speed limit (100km/h)',
    9: 'Speed limit (120km/h)',
    10: 'No passing',
    11: 'No passing veh over 3.5 tons',
    12: 'Right-of-way at intersection',
    13: 'Priority road',
    14: 'Yield',
    15: 'Stop',
    16: 'No vehicles',
    17: 'Veh > 3.5 tons prohibited',
    18: 'No entry',
    19: 'General caution',
    20: 'Dangerous curve left',
    21: 'Dangerous curve right',
    22: 'Double curve',
    23: 'Bumpy road',
    24: 'Slippery road',
    25: 'Road narrows on the right',
    26: 'Road work',
    27: 'Traffic signals',
    28: 'Pedestrians',
    29: 'Children crossing',
    30: 'Bicycles crossing',
    31: 'Beware of ice/snow',
    32: 'Wild animals crossing',
    33: 'End speed + passing limits',
    34: 'Turn right ahead',
    35: 'Turn left ahead',
    36: 'Ahead only',
    37: 'Go straight or right',
    38: 'Go straight or left',
    39: 'Keep right',
    40: 'Keep left',
    41: 'Roundabout mandatory',
    42: 'End of no passing',
    43: 'End no passing veh > 3.5 tons'
}

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Traffic Sign Classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Function to classify an image
def classify(image):
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict(image)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred) * 100
    sign = classes[pred_class + 1]
    return sign, confidence

# Function to process video or camera
def process_video(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Classify the frame
        sign, confidence = classify(frame)

        # Display the result on the frame
        cv2.putText(frame, f"{sign} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Traffic Sign Classification", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to open video file
def open_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if file_path:
        process_video(file_path)

# Function to open camera
def open_camera():
    process_video(0)  # 0 is the default camera

# Buttons for video and camera
video_button = Button(top, text="Open Video File", command=open_video_file, padx=10, pady=5)
video_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
video_button.place(relx=0.1, rely=0.1)

camera_button = Button(top, text="Open Camera", command=open_camera, padx=10, pady=5)
camera_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
camera_button.place(relx=0.3, rely=0.1)

# Main loop
heading = Label(top, text="Traffic Sign Classification", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()