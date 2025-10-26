import os
import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

# ------------------------------
# TensorFlow 로그 숨기기
# ------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------------------
# 설정
# ------------------------------
max_num_hands = 1
gesture = {i: chr(97+i) for i in range(26)}
gesture[26] = 'spacing'
gesture[27] = 'clear'
recognizeDelay = 1  # 글자 인식 지연

# ------------------------------
# Mediapipe 설정
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------
# 학습 데이터 로드
# ------------------------------
file = np.genfromtxt('dataSet.txt', delimiter=',')
angleFile = file[:, :-1]
labelFile = file[:, -1]
angle_train = angleFile.astype(np.float32)
label_train = labelFile.astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(angle_train, cv2.ml.ROW_SAMPLE, label_train)

# ------------------------------
# 카메라 연결 (전체 화면)
# ------------------------------
window_name = "HandTracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

sentence = ''
prev_index = -1
startTime = time.time()

# ------------------------------
# UI 그리는 함수
# ------------------------------
def draw_ui(img, gesture_text, sentence, landmarks=None):
    h, w, _ = img.shape

    # 하단 반투명 박스
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
    alpha = 0.5
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 상단 중앙: 현재 제스처
    cv2.putText(img, f'Gesture: {gesture_text.upper()}',
                (w//2 - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    # 하단: 누적 문장
    cv2.putText(img, sentence,
                (20, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # 손 랜드마크 표시
    if landmarks is not None:
        for lm in landmarks.landmark:
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)
        mp_drawing.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,200,0), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0,150,0), thickness=2))

    return img

# ------------------------------
# 메인 루프
# ------------------------------
while True:
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)  # 1 = 좌우, 0 = 상하, -1 = 좌우+상하

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(imgRGB)

    gesture_text = ''
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # 손가락 각도 계산
            v1 = joint[[0,1,2,3,5,6,7,9,10,11,13,14,15,17,18], :]
            v2 = joint[[1,2,3,4,6,7,8,10,11,12,14,15,16,18,19], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            compareV1 = v
            compareV2 = np.roll(v, -1, axis=0)[:15, :]
            angle_test = np.arccos(np.einsum('nt,nt->n', compareV1[:15], compareV2))
            angle_test = np.degrees(angle_test).astype(np.float32)

            # a 키: 테스트 데이터 저장
            if keyboard.is_pressed('a'):
                with open('test.txt', 'a') as f:
                    f.write(','.join(map(str, angle_test)) + ',27\n')
                    print("저장 완료!")

            # KNN 예측
            if len(angle_test) == angle_train.shape[1]:
                data = angle_test.reshape(1, -1).astype(np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, k=3)
                index = int(results[0][0])
                gesture_text = gesture.get(index, '')

                # 글자 인식
                if prev_index != index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 26:
                            sentence += ' '
                        elif index == 27:
                            sentence = ''
                        else:
                            sentence += gesture.get(index, '')
                        startTime = time.time()
            
            img = draw_ui(img, gesture_text, sentence, res)

    cv2.imshow(window_name, img)

    if cv2.waitKey(1) & 0xFF == ord('b'):  # b 키: 종료
        break

cap.release()
cv2.destroyAllWindows()
