# import cv2
# import face_recognition
# import os

# known_faces = []
# known_names = []

# for img_name in os.listdir('detected'):
#     img_path = os.path.join('detected', img_name)
#     img = face_recognition.load_image_file(img_path)
#     encoding = face_recognition.face_encodings(img)[0]
#     known_faces.append(encoding)
#     known_names.append(os.path.splitext(img_name)[0])

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Find faces in the current frame
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)

#     # Loop through each face in the current frame
#     best_match = {"name": None, "accuracy": 0}
#     category = "none"

#     for face_encoding, face_location in zip(face_encodings, face_locations):
#         # Compare the current face encoding to the known faces
#         matches = face_recognition.compare_faces(known_faces, face_encoding)

#         # Find the highest accuracy match
#         accuracies = face_recognition.face_distance(known_faces, face_encoding)
#         best_index = accuracies.argmin()
#         name = known_names[best_index]
#         accuracy = (1 - accuracies[best_index]) * 100

#         # Update the best match if this match is better
#         if accuracy > best_match["accuracy"] and accuracy > 50:
#             best_match["name"] = name
#             best_match["accuracy"] = accuracy
#         else:
#             best_match["name"] = "unknown"
#             best_match["accuracy"] = accuracy

#         # Draw a box and label around the face
#         y1, x2, y2, x1 = face_location
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{best_match['name']} ({best_match['accuracy']:.2f}%)", (
#             x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     # Display the name with the highest accuracy
#     cv2.putText(frame, f"{best_match['name']} ({best_match['accuracy']:.2f}%)", (
#         10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     # Display the current frame
#     cv2.imshow('Face Recognition', frame)

#     # Quit the program if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import face_recognition
import os

known_faces = []
known_names = []
known_categories = []

for img_name in os.listdir('detected'):
    img_path = os.path.join('detected', img_name)
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)[0]
    name, category = os.path.splitext(img_name)[0].split('-')
    known_faces.append(encoding)
    known_names.append(name)
    known_categories.append(category)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find faces in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in the current frame
    best_match = {"name": None, "accuracy": 0, "category": None}

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the current face encoding to the known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)

        # Find the highest accuracy match
        accuracies = face_recognition.face_distance(known_faces, face_encoding)
        best_index = accuracies.argmin()
        name = known_names[best_index]
        accuracy = (1 - accuracies[best_index]) * 100
        category = known_categories[best_index]

        # Update the best match if this match is better
        if accuracy > best_match["accuracy"] and accuracy > 50:
            best_match["name"] = name
            best_match["accuracy"] = accuracy
            best_match["category"] = category
        else:
            best_match["name"] = "unknown"
            best_match["accuracy"] = accuracy
            best_match["category"] = "none"

        # Draw a box and label around the face
        y1, x2, y2, x1 = face_location
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{best_match['name']}", (
            x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the name with the highest accuracy
    if accuracy > 50:
        cv2.putText(frame, f"Name: {best_match['name']}", (
            10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Percent Accuracy: {best_match['accuracy']:.2f}%", (
            10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Category: {best_match['category']})", (
            10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Name: {best_match['name']}", (
            10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the current frame
    cv2.imshow('Face Recognition', frame)

    # Quit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
