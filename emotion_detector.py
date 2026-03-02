# Multiple faces

# The Model have been trained on FER-2013 dataset available on Kaggle 
# https://www.kaggle.com/datasets/msambare/fer2013

import cv2
import pathlib
import numpy as np
from model import *
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--record', action='store_true')
parser.add_argument('-fn', '--filename', type=str, default='output.avi')

args = parser.parse_args()



# define the face detector and the camera reader
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))
cam = cv2.VideoCapture(0)
framerate = cam.get(cv2.CAP_PROP_FPS)

cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 48)

ret, frame = cam.read()



frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# if a record is required
if args.record:
    writer = cv2.VideoWriter(f"{args.filename}.avi", fourcc, 18.0, (frame_width, frame_height))


# some image manipulation function
# model have been trained on grayscale images 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


classes = ["angry","disgust","fear","happy","neutral","sad", "surprise"]


# initialize and load the model
my_model = Net()
my_model.load_state_dict(torch.load("model3.pkl", weights_only=True))
my_model.eval()


while True:
    ret, frame = cam.read()

    if not(ret):
        continue
        
    gray_frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)

    faces = clf.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48,48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    rectangles = []
    new_faces = []

    for (x, y, width, height) in faces:
        rectangles.append([(x,y), (x+width, y+height), (0,0,255), 3])
    
    for rectangle in rectangles:
        (x1,y1), (x2, y2), color, thickness = rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        face_roi = frame[y1:y2, x1:x2]
        resized_frame = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        new_faces.append(transform(Image.fromarray(resized_frame)))

    new_faces = torch.from_numpy(np.array(new_faces))

    sc = []    
    with torch.no_grad():
        for face in new_faces:
            face = face.unsqueeze(dim=0)
            score = my_model(face)
            sc.append(score)

    moods = []
    for s in sc: #sc = scores
        mood = torch.argmax(s)
        moods.append(classes[mood])

    for mood, rect in list(zip(moods, rectangles)):
        (x1, y1), (x2, y2), _, _ = rect
        cv2.putText(frame, f"{mood}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if args.record:
        writer.write(frame)

    cv2.imshow('Camera', frame)



    if cv2.waitKey(1) == ord('q'):
        break

cam.release()

if args.record:
    writer.release()
cv2.destroyAllWindows()