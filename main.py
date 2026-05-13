import cv2
import mediapipe as mp
import time

# 1. MediaPipe Modüllerinin Başlatılması
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Yüz ve El modellerini yapılandırıyoruz
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

# Kamerayı başlat (Genellikle 0 veya 1)
cap = cv2.VideoCapture(0)

pTime = 0 # FPS hesabı için önceki zaman

print("Sistem başlatılıyor... Çıkmak için 'q' tuşuna basın.")

while True:
    success, img = cap.read()
    if not success:
        print("Kameradan görüntü alınamadı!")
        break

    # OpenCV görüntüyü BGR okur, MediaPipe ise RGB bekler. Rengi dönüştürüyoruz.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Yüz Tespiti (Adım 1)
    face_results = face_detection.process(imgRGB)
    face_detected = False

    if face_results.detections:
        face_detected = True
        for detection in face_results.detections:
            mp_drawing.draw_detection(img, detection)

    # 3. El Tespiti ve Landmark (Eklem Noktası) Çıkarımı
    hand_results = hands.process(imgRGB)

    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            # Elin iskeletini ekrana çizdiriyoruz
            mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            # İleride her bir eklem noktasının (0'dan 20'ye kadar) (x, y) koordinatlarını buradan çekeceğiz
            # Örneğin işaret parmağının ucu (Landmark 8):
            # ix, iy = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)

    # Sistem Durumu Bilgisi Yazdırma
    status_text = "Yuz: BULUNDU" if face_detected else "Yuz: BEKLENIYOR"
    color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.putText(img, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # FPS Hesaplama (Mühendislik projelerinde gecikme (latency) analizi için önemlidir)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Air-Pass Security System - Beyin Testi", img)

    # 'q' tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik işlemleri
cap.release()
cv2.destroyAllWindows()