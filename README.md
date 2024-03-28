# EmotionDrive-Emotionserkennung-f-r-sicheres-Fahren
EmotionDrive überwacht die Emotionen des Fahrers mittels Gesichtserkennung und warnt bei Anzeichen von Müdigkeit, Ablenkung oder Stress, um die Fahrsicherheit zu erhöhen.
import cv2
from emotion_detector import EmotionDetector # This is a hypothetical library/API

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def check_driver_emotion(frame):
    """
    Detect the driver's face and classify the emotion.
    Returns the detected emotion.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        emotion = EmotionDetector.detect_emotion(face)
        return emotion

    return "No Face Detected"

def main():
    while True:
        _, frame = cap.read()

        emotion = check_driver_emotion(frame)

        if emotion in ["Fatigue", "Distraction", "Stress"]:
            print(f"Warning: {emotion} detected. Please take a break or focus on the road.")

        # Display the resulting frame
        cv2.imshow('EmotionDrive Monitoring', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
