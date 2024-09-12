import cv2
import pyautogui
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera")
    exit()

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utility from MediaPipe
mp_drawing = mp.solutions.drawing_utils

def detect_gesture(hand_landmarks):
    """
    Detect gestures based on the hand landmarks.
    """
    if hand_landmarks is None:
        return 'none'

    # Get the y-coordinate of thumb and index fingertips
    thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

    # Simple gesture detection based on relative positions
    if index_finger_y < thumb_y:
        return 'pointing up'
    elif index_finger_y > thumb_y:
        return 'pointing down'
    else:
        return 'other'

while True:
    ret, frame = cap.read()

    if not ret:
        print('Error: Failed to grab frame.')
        break

    # Resize and process the image
    frame_resized = cv2.resize(frame, (640, 480))
    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(image_rgb)

    # If hand landmarks are detected, process gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture
            gesture = detect_gesture(hand_landmarks)
            print(f"Gesture detected: {gesture}")

            # Perform actions based on the gesture (example: pointing up triggers a keyboard action)
            if gesture == 'pointing up':
                pyautogui.press('up')  # Simulate pressing the 'up' key
            elif gesture == 'pointing down':
                pyautogui.press('down')  # Simulate pressing the 'down' key

    # Display the frame with drawn landmarks
    cv2.imshow('Hand Gesture Recognition', frame_resized)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
