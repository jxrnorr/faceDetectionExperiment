import cv2

def extract_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        return face
    else:
        return None


def compare_faces(image1_path, image2_path):
    face1 = extract_face(image1_path)
    face2 = extract_face(image2_path)
    
    if face1 is None or face2 is None:
        return "Error: Unable to detect face in one or both images", None
    
    # Resize images to a common size
    face1_resized = cv2.resize(face1, (100, 100))  # Adjust the size as needed
    face2_resized = cv2.resize(face2, (100, 100))  # Adjust the size as needed
    
    diff = cv2.absdiff(face1_resized, face2_resized)
    similarity = 1 - (diff/255.0).mean()
    
    # You can adjust this threshold according to your requirements
    threshold = 0.9
    
    if similarity > threshold:
        return "Same person", similarity
    else:
        return "Different persons", similarity



if __name__ == "__main__":
    image1_path = "/Users/User/OneDrive/Desktop/testphoto/img5.jpg"
    image2_path = "/Users/User/OneDrive/Desktop/testphoto/img6.jpg"
    image3_path = "/Users/User/OneDrive/Desktop/testphoto/img7.jpg"
    image4_path = "/Users/User/OneDrive/Desktop/testphoto/img8.jpg"
    image5_path = "/Users/User/OneDrive/Desktop/testphoto/img9.jpg"
    image6_path = "/Users/User/OneDrive/Desktop/testphoto/img10.jpg"
    
    # Compare image1 and image2
    result1_2, similarity1_2 = compare_faces(image1_path, image2_path)
    print("Comparison between image 1 and image 2:", result1_2)
    print("Similarity between image 1 and image 2:", similarity1_2)
    
    # Compare image1 and image3
    result1_3, similarity1_3 = compare_faces(image1_path, image3_path)
    print("Comparison between image 1 and image 3:", result1_3)
    print("Similarity between image 1 and image 3:", similarity1_3)
    
    # Compare image1 and image2
    result1_4, similarity1_4 = compare_faces(image1_path, image4_path)
    print("Comparison between image 1 and image 4:", result1_4)
    print("Similarity between image 1 and image 4:", similarity1_4)

    # Compare image2 and image3
    result4_5, similarity4_5 = compare_faces(image4_path, image5_path)
    print("Comparison between image 4 and image 5:", result4_5)
    print("Similarity between image 4 and image 5:", similarity4_5)

    # Compare image1 and image2
    result4_6, similarity4_6 = compare_faces(image4_path, image6_path)
    print("Comparison between image 4 and image 6:", result4_6)
    print("Similarity between image 4 and image 6:", similarity4_6)

