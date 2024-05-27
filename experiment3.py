import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/User/OneDrive/Desktop/testphoto/shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects = detector(gray, 1)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)

        # Convert the landmark (x, y)-coordinates to a NumPy array
        landmarks = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
        
    return landmarks

# Paths to the images
image_paths = [
    "/Users/User/OneDrive/Desktop/testphoto/img1.jpg",
    "/Users/User/OneDrive/Desktop/testphoto/img2.jpg",
    "/Users/User/OneDrive/Desktop/testphoto/img3.jpg",
    "/Users/User/OneDrive/Desktop/testphoto/img4.jpg"
]

# Load the images
images = [cv2.imread(path) for path in image_paths]

# Extract landmarks
landmarks = [get_landmarks(image) for image in images]

# Compare each image with every other image
for i in range(len(images)):
    for j in range(i+1, len(images)):
        dist = distance.euclidean(landmarks[i].flatten(), landmarks[j].flatten())
        print(f'Distance between image{i+1} and image{j+1}: {dist}')

