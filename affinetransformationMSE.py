import dlib
import cv2
import numpy as np
import mediapipe as mp
import os
from skimage.transform import estimate_transform, AffineTransform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Initialize the facial landmark detector from Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize the facial landmark detector from dlib
predictor = dlib.shape_predictor("/Users/User/OneDrive/Desktop/testphoto/shape_predictor_68_face_landmarks.dat")

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using dlib
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 1)
    
    if len(faces) == 0:
        return None
    
    # Assuming only one face detected
    face = faces[0]
    
    # Predict facial landmarks using dlib
    landmarks = predictor(gray, face)
    
    # Convert landmarks to numpy array
    landmark_points = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    return landmark_points

def affine_transform_landmarks(landmarks1, landmarks2):
    # Estimate affine transformation
    transform = estimate_transform('affine', landmarks1, landmarks2)
    # Apply the affine transformation to landmarks1
    landmarks1_transformed = transform(landmarks1)
    return landmarks1_transformed

def calculate_similarity_mse(landmarks1, landmarks2):
    # Align the landmarks using affine transformation
    landmarks1_aligned = affine_transform_landmarks(landmarks1, landmarks2)
    
    # Calculate Mean Squared Error between corresponding aligned landmarks
    mse = np.mean(np.square(landmarks1_aligned - landmarks2))
    
    # Convert MSE to similarity score
    similarity_score = 1 / (1 + mse)  # Higher MSE means lower similarity
    
    return similarity_score

def compare_faces_with_landmarks(image1_path, image2_path):
    # Extract landmarks from both images
    landmarks1 = extract_landmarks(image1_path)
    landmarks2 = extract_landmarks(image2_path)
    
    if landmarks1 is None or landmarks2 is None:
        return "Error: Unable to detect face in one or both images"
    
    # Calculate similarity score based on differences in landmark positions
    similarity_score = calculate_similarity_mse(landmarks1, landmarks2)
    
    # You can adjust the threshold based on your requirements
    threshold = 0.4  # Adjust this threshold as needed
    
    if similarity_score > threshold:
        return "Same person", similarity_score
    else:
        return "Different persons", similarity_score

if __name__ == "__main__":
    image1_path = "/Users/User/OneDrive/Desktop/testphoto/img1.jpg"
    image2_path = "/Users/User/OneDrive/Desktop/testphoto/img2.jpg"
    image3_path = "/Users/User/OneDrive/Desktop/testphoto/img3.jpg"
    image4_path = "/Users/User/OneDrive/Desktop/testphoto/img4.jpg"
    
    # Compare image1 and image2
    result1_2, similarity1_2 = compare_faces_with_landmarks(image1_path, image2_path)
    print("Comparison between image 1 and image 2:", result1_2)
    print("Similarity between image 1 and image 2:", similarity1_2)
    
    # Compare image1 and image3
    result1_3, similarity1_3 = compare_faces_with_landmarks(image1_path, image3_path)
    print("Comparison between image 1 and image 3:", result1_3)
    print("Similarity between image 1 and image 3:", similarity1_3)
    
    # Compare image1 and image4
    result1_4, similarity1_4 = compare_faces_with_landmarks(image1_path, image4_path)
    print("Comparison between image 1 and image 4:", result1_4)
    print("Similarity between image 1 and image 4:", similarity1_4)

    # Compare image2 and image3
    result2_3, similarity2_3 = compare_faces_with_landmarks(image2_path, image3_path)
    print("Comparison between image 2 and image 3:", result2_3)
    print("Similarity between image 2 and image 3:", similarity2_3)

    # Compare image2 and image4
    result2_4, similarity2_4 = compare_faces_with_landmarks(image2_path, image4_path)
    print("Comparison between image 2 and image 4:", result2_4)
    print("Similarity between image 2 and image 4:", similarity2_4)

    # Compare image3 and image4
    result3_4, similarity3_4 = compare_faces_with_landmarks(image3_path, image4_path)
    print("Comparison between image 3 and image 4:", result3_4)
    print("Similarity between image 3 and image 4:", similarity3_4)
