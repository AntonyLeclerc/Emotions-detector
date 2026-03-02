# Emotions Detector

Computer vision project.  
Based on a Convolutional Neural Network.

My model (mymodel.pkl) has been trained on the FER-2013 dataset, consisting of:  
  - 28.709 train images
  - 7.178 test images

Scattered among 7 different classes: Angry / Disgust / Fear / Happy / Neutral / Sad / Surprise

---

For inference, I am using my opencv-python and a VideoCapture object, then turning the image into grayscale and resizing it to a 48x48 dimension (image dimensions in the FER-2013 dataset).

