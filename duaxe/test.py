import cv2
import mediapipe as mp

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Mở camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    # Chuyển đổi hình ảnh sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _  = img_rgb.shape
    # Vẽ các điểm mốc trên bàn tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Lấy tọa độ các điểm mốc của ngón cái
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            thumb_mcp = hand_landmarks.landmark[2]
            thumb_cmc = hand_landmarks.landmark[1]
            index_mcp = hand_landmarks.landmark[5]


            # Kiểm tra điều kiện để xác định cử chỉ giơ ngón cái
            if (h*(thumb_cmc.y-thumb_mcp.y)>35) and (h*(thumb_mcp.y-thumb_ip.y)>35):
                cv2.putText(img, 'Thumbs Up', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow('Hand Tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

