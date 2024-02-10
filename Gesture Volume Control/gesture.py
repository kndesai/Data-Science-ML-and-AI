import cv2
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Initialize the pycaw volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Define the maximum distance that corresponds to 100% volume (needs to be calibrated)
MAX_HAND_DISTANCE = 0.2  # Adjust this value based on your own maximum stretch

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally and convert the color space from BGR to RGB
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image)

    # Draw hand landmarks
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks for thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate distance between thumb tip and index finger tip
            current_distance = math.sqrt(
                (thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2)

            # Scale the distance to the maximum hand distance
            scaled_distance = min(current_distance / MAX_HAND_DISTANCE, 1.0)

            # Convert scaled distance to volume
            vol = scaled_distance * (volume.GetVolumeRange()[1] - volume.GetVolumeRange()[0]) + volume.GetVolumeRange()[
                0]

            # Set the system volume
            volume.SetMasterVolumeLevel(vol, None)

            # Display the volume level on the webcam window
            vol_percent = int(scaled_distance * 100)
            cv2.putText(image, f'Volume: {vol_percent}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

