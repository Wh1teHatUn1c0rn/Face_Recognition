import cv2
import face_recognition


class CameraFaceRecognition:
    def __init__(self, camera_url, known_faces):
        self.camera_url = camera_url
        self.known_faces = known_faces

    def start(self):
        # Connect to the camera
        cap = cv2.VideoCapture(self.camera_url)

        while True:
            # Capture a frame from the camera
            ret, frame = cap.read()

            # Convert the frame to a format suitable for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)

            # Loop through the detected faces
            for top, right, bottom, left in face_locations:
                # Crop the face from the frame
                face_image = rgb_frame[top:bottom, left:right]

                # Encode the face
                face_encoding = face_recognition.face_encodings(face_image)[0]

                # Compare the face to the known faces
                match = face_recognition.compare_faces(self.known_faces, face_encoding)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # If the face is a match, put the name of the person
                if True in match:
                    name = "Known Person"
                else:
                    name = "Unknown Person"

                # Put the name on the frame
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                # Show the frame
                cv2.imshow("Camera", frame)

                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the camera and destroy the window
            cap.release()
            cv2.destroyAllWindows()
