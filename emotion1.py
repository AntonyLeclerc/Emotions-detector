# One face at a time

#from model import *
import cv2
import pathlib
import numpy as np
from model import *
from PIL import Image


# define the face detector and the camera reader
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 48)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# some image manipulation function
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

classes = ["angry","disgust","fear","happy","neutral","sad", "surprise"]


# initialize and load the model


my_model = Net()
my_model.load_state_dict(torch.load("mymodel_weigths.pkl", weights_only=True))
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


    #rectangles = []
    for (x, y, width, height) in faces:
        #rectangles.append([(x,y), (x+width, y+height), (0,0,255), 3])
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0,0,255), 3)

    face_roi = frame[y:y+height, x:x+width]
    resized_frame = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

    img_pred = transform(Image.fromarray(resized_frame))
    img_pred = img_pred.unsqueeze(dim=0)

    with torch.no_grad():
        scores = my_model(img_pred)

    am = torch.argmax(scores)
    mood = classes[am]

    cv2.putText(frame, f"{mood}", (x, y+height+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Camera', frame)


    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()