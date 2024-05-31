from deepface import DeepFace

# Load and analyze images
result1 = DeepFace.analyze("/Users/User/OneDrive/Desktop/testphoto/img5.jpg")
result2 = DeepFace.analyze("/Users/User/OneDrive/Desktop/testphoto/img6.jpg")
result3 = DeepFace.analyze("/Users/User/OneDrive/Desktop/testphoto/img7.jpg")
result4 = DeepFace.analyze("/Users/User/OneDrive/Desktop/testphoto/img8.jpg")
result5 = DeepFace.analyze("/Users/User/OneDrive/Desktop/testphoto/img9.jpg")
result6 = DeepFace.analyze("/Users/User/OneDrive/Desktop/testphoto/img10.jpg")

# Compare images
comparison_result1 = DeepFace.verify("/Users/User/OneDrive/Desktop/testphoto/img5.jpg", 
                                    "/Users/User/OneDrive/Desktop/testphoto/img6.jpg")

if comparison_result1['verified']:
    print(f"Image 1 and Image 2 are similar with a confidence: {comparison_result1['distance']}")
else:
    print("Image 1 and Image 2 are not similar")

comparison_result2 = DeepFace.verify("/Users/User/OneDrive/Desktop/testphoto/img5.jpg", 
                                    "/Users/User/OneDrive/Desktop/testphoto/img7.jpg")

if comparison_result2['verified']:
    print(f"Image 1 and Image 3 are similar with a confidence: {comparison_result2['distance']}")
else:
    print("Image 1 and Image 3 are not similar")

comparison_result3 = DeepFace.verify("/Users/User/OneDrive/Desktop/testphoto/img5.jpg", 
                                    "/Users/User/OneDrive/Desktop/testphoto/img8.jpg")

if comparison_result3['verified']:
    print(f"Image 1 and Image 4 are similar with a confidence: {comparison_result3['distance']}")
else:
    print("Image 1 and Image 4 are not similar")

comparison_result4 = DeepFace.verify("/Users/User/OneDrive/Desktop/testphoto/img8.jpg", 
                                    "/Users/User/OneDrive/Desktop/testphoto/img9.jpg")

if comparison_result4['verified']:
    print(f"Image 4 and Image 5 are similar with a confidence: {comparison_result4['distance']}")
else:
    print("Image 4 and Image 5 are not similar")

comparison_result5 = DeepFace.verify("/Users/User/OneDrive/Desktop/testphoto/img8.jpg", 
                                    "/Users/User/OneDrive/Desktop/testphoto/img10.jpg")

if comparison_result5['verified']:
    print(f"Image 4 and Image 6 are similar with a confidence: {comparison_result5['distance']}")
else:
    print("Image 4 and Image 6 are not similar")

comparison_result6 = DeepFace.verify("/Users/User/OneDrive/Desktop/testphoto/img9.jpg", 
                                    "/Users/User/OneDrive/Desktop/testphoto/img10.jpg")

if comparison_result6['verified']:
    print(f"Image 5 and Image 6 are similar with a confidence: {comparison_result6['distance']}")
else:
    print("Image 5 and Image 6 are not similar")
