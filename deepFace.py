from deepface import DeepFace

# Load and analyze images
result1 = DeepFace.analyze("/Users/User/OneDrive/Desktop/testphoto/img5.jpg")
result2 = DeepFace.analyze("/Users/User/OneDrive/Desktop/testphoto/img7.jpg")

# Compare images
comparison_result = DeepFace.verify("/Users/User/OneDrive/Desktop/testphoto/img5.jpg", 
                                    "/Users/User/OneDrive/Desktop/testphoto/img7.jpg")

if comparison_result['verified']:
    print(f"Images are similar with a confidence: {comparison_result['distance']}")
else:
    print("Images are not similar")
