import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time

# ------------------------------
# 설정
# ------------------------------
max_num_hands = 1
gesture = {
    0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',
    8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',
    15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',
    22:'w',23:'x',24:'y',25:'z',26:'spacing',27:'clear'
}

recognizeDelay = 1  # 글자 인식 지연

# ------------------------------
# Mediapipe 설정
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------
# 학습 데이터 로드
# ------------------------------
file = np.genfromtxt('dataSet.txt', delimiter=',')
angleFile = file[:, :-1]   # feature 15개
labelFile = file[:, -1]    # 레이블
angle_train = angleFile.astype(np.float32)
label_train = labelFile.astype(np.float32)

# KNN 학습
knn = cv2.ml.KNearest_create()
knn.train(angle_train, cv2.ml.ROW_SAMPLE, label_train)

# ------------------------------
# 카메라 연결
# ------------------------------
cap = cv2.VideoCapture(0)
sentence = ''
prev_index = -1
startTime = time.time()

# ------------------------------
# 메인 루프
# ------------------------------
while True:
    ret, img = cap.read()
    if not ret:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # --------------------------
            # 손가락 각도 계산
            # --------------------------
            v1 = joint[[0,1,2,3,5,6,7,9,10,11,13,14,15,17,18], :]
            v2 = joint[[1,2,3,4,6,7,8,10,11,12,14,15,16,18,19], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            compareV1 = v
            compareV2 = np.roll(v, -1, axis=0)[:15, :]  # 15개 feature 맞춤
            angle_test = np.arccos(np.einsum('nt,nt->n', compareV1[:15], compareV2))
            angle_test = np.degrees(angle_test).astype(np.float32)

            # --------------------------
            # a 키 누르면 데이터 저장
            # --------------------------
            if keyboard.is_pressed('a'):
                with open('test.txt', 'a') as f:
                    f.write(','.join(map(str, angle_test)) + ',27\n')  # 27 = clear
                    print("저장 완료!")

            # --------------------------
            # KNN 예측
            # --------------------------
            if len(angle_test) == angle_train.shape[1]:  # feature 수 체크
                data = angle_test.reshape(1, -1).astype(np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, k=3)
                index = int(results[0][0])

                # ----------------------
                # 글자 인식
                # ----------------------
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

                # 화면에 제스처 표시
                cv2.putText(img, gesture.get(index, '').upper(),
                            (int(res.landmark[0].x * img.shape[1] - 10),
                             int(res.landmark[0].y * img.shape[0] + 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # --------------------------
            # 랜드마크 그리기
            # --------------------------
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            cv2.putText(img, sentence, (20, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    cv2.imshow('HandTracking', img)
    if cv2.waitKey(1) == ord('b'):  # b 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
