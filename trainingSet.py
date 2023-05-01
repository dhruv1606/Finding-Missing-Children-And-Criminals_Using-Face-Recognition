import cv2

# Set up the camera
cap = cv2.VideoCapture(0)

# Load the face detection algorithm
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Get the name of the person to save the image
name = input("Enter the name of the person: ")
category = input("Enter the category of the person: ")
if category=="Criminal" : 
    age = input("Enter the age of the criminal: ")
else:
    age = input("Enter the age of the missing child: ")
    

# Set up the file name
filename = f"detected/{name}-{category}-{age}.jpg"

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # If a face is detected, save the image
    if len(faces) > 0:
        # Draw a rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the image to a file
        cv2.imwrite(filename, frame)
        print(f"Image of {name} saved as {filename}")
        break

    # Display the frame
    cv2.imshow('Capture', frame)

    # Quit the program if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
