from flask import Flask, render_template, Response, request, redirect
import cv2
import face_recognition
import os
import sys

app = Flask(__name__)

known_faces = []
known_names = []
known_categories = []
known_ages = []

for img_name in os.listdir('detected'):
    img_path = os.path.join('detected', img_name)
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)[0]
    name, category, age = os.path.splitext(img_name)[0].split('-')
    known_faces.append(encoding)
    known_names.append(name)
    known_categories.append(category)
    known_ages.append(age)

camera = cv2.VideoCapture(0)


def detect_faces():
    best_match = {"name": None, "accuracy": 0, "category": None, "age": None}
    while True:
        ret, frame = camera.read()

        if not ret:
            break

        # Find faces in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)


        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the current face encoding to the known faces
            matches = face_recognition.compare_faces(
                known_faces, face_encoding)

            # Find the highest accuracy match
            accuracies = face_recognition.face_distance(
                known_faces, face_encoding)
            best_index = accuracies.argmin()
            name = known_names[best_index]
            accuracy = (1 - accuracies[best_index]) * 100
            category = known_categories[best_index]
            age = known_ages[best_index]

            # Update the best match if this match is better
            if accuracy > best_match["accuracy"] and accuracy > 50:
                best_match["name"] = name
                best_match["accuracy"] = accuracy
                best_match["category"] = category
                best_match["age"] = age
            else:
                best_match["name"] = "unknown"
                best_match["accuracy"] = accuracy
                best_match["category"] = "none"
                best_match["age"] = "unknown"

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
                10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Category: {best_match['category']}", (
                10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {best_match['age']}", (
                10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # return best_match[name]
        else:
            cv2.putText(frame, f"Name: {best_match['name']}", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)

        # Send the frame to the browser as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/camera')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detectedperson')
def detectedperson():
    name = request.args.get('name', default='unknown')
    return render_template('person.html', name)


if __name__ == '__main__':
    app.run(debug=True)
