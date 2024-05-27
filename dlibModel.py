import cv2
import dlib
import numpy as np

# Load detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_descriptor(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    descriptors = []

    for face in faces:
        shape = predictor(gray, face)
        descriptor = face_rec_model.compute_face_descriptor(image, shape)
        descriptors.append(np.array(descriptor))

    return descriptors

# Get face descriptors for the images
descriptors1 = get_face_descriptor("/Users/User/OneDrive/Desktop/testphoto/img5.jpg")
descriptors2 = get_face_descriptor("/Users/User/OneDrive/Desktop/testphoto/img6.jpg")

# Compare faces
for i, d1 in enumerate(descriptors1):
    for j, d2 in enumerate(descriptors2):
        dist = np.linalg.norm(d1 - d2)
        if dist < 0.6:  # You can adjust the threshold
            print(f"Image 1 and Image 2: Face {i+1} and Face {j+1} Similarity: {1 - dist}")
