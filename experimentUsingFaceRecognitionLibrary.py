import face_recognition

# Load images
image1 = face_recognition.load_image_file("/Users/User/OneDrive/Desktop/testphoto/img5.jpg")
image2 = face_recognition.load_image_file("/Users/User/OneDrive/Desktop/testphoto/img6.jpg")
image3 = face_recognition.load_image_file("/Users/User/OneDrive/Desktop/testphoto/img7.jpg")
image4 = face_recognition.load_image_file("/Users/User/OneDrive/Desktop/testphoto/img8.jpg")
image5 = face_recognition.load_image_file("/Users/User/OneDrive/Desktop/testphoto/img9.jpg")
image6 = face_recognition.load_image_file("/Users/User/OneDrive/Desktop/testphoto/img10.jpg")


# Find face landmarks
face_landmarks_image1 = face_recognition.face_landmarks(image1)
face_landmarks_image2 = face_recognition.face_landmarks(image2)
face_landmarks_image3 = face_recognition.face_landmarks(image3)
face_landmarks_image4 = face_recognition.face_landmarks(image4)
face_landmarks_image5 = face_recognition.face_landmarks(image5)
face_landmarks_image6 = face_recognition.face_landmarks(image6)

# Encode face landmarks
encodings_image1 = face_recognition.face_encodings(image1)
encodings_image2 = face_recognition.face_encodings(image2)
encodings_image3 = face_recognition.face_encodings(image3)
encodings_image4 = face_recognition.face_encodings(image4)
encodings_image5 = face_recognition.face_encodings(image5)
encodings_image6 = face_recognition.face_encodings(image6)

# Compare faces
for i, encoding1 in enumerate(encodings_image1):
    for j, encoding2 in enumerate(encodings_image2):
        match = face_recognition.compare_faces([encoding1], encoding2, tolerance=0.6)
        if match[0]:
            distance = face_recognition.face_distance([encoding1], encoding2)[0]
            print(f"Image 1 and Image 2: Face {i+1} and Face {j+1} Similarity: {1 - distance}")

for i, encoding1 in enumerate(encodings_image1):
    for j, encoding3 in enumerate(encodings_image3):
        match = face_recognition.compare_faces([encoding1], encoding3, tolerance=0.6)
        if match[0]:
            distance = face_recognition.face_distance([encoding1], encoding3)[0]
            print(f"Image 1 and Image 3: Face {i+1} and Face {j+1} Similarity: {1 - distance}")

for i, encoding1 in enumerate(encodings_image1):
    for j, encoding4 in enumerate(encodings_image4):
        match = face_recognition.compare_faces([encoding1], encoding4, tolerance=0.6)
        if match[0]:
            distance = face_recognition.face_distance([encoding1], encoding4)[0]
            print(f"Image 1 and Image 4: Face {i+1} and Face {j+1} Similarity: {1 - distance}")

for i, encoding3 in enumerate(encodings_image1):
    for j, encoding4 in enumerate(encodings_image4):
        match = face_recognition.compare_faces([encoding3], encoding4, tolerance=0.6)
        if match[0]:
            distance = face_recognition.face_distance([encoding3], encoding4)[0]
            print(f"Image 3 and Image 4: Face {i+1} and Face {j+1} Similarity: {1 - distance}")

for i, encoding4 in enumerate(encodings_image4):
    for j, encoding5 in enumerate(encodings_image5):
        match = face_recognition.compare_faces([encoding4], encoding5, tolerance=0.6)
        if match[0]:
            distance = face_recognition.face_distance([encoding4], encoding5)[0]
            print(f"Image 4 and Image 5: Face {i+1} and Face {j+1} Similarity: {1 - distance}")

for i, encoding4 in enumerate(encodings_image4):
    for j, encoding6 in enumerate(encodings_image6):
        match = face_recognition.compare_faces([encoding4], encoding6, tolerance=0.6)
        if match[0]:
            distance = face_recognition.face_distance([encoding4], encoding6)[0]
            print(f"Image 4 and Image 6: Face {i+1} and Face {j+1} Similarity: {1 - distance}")

# Repeat the above comparisons for other image pairs as needed
