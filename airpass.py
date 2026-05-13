import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import serial
import os
import urllib.request

# --- AYARLAR ---
ENABLE_SERIAL = False # Arduino bağlıysa True yapın
SERIAL_PORT = 'COM3'  # Arduino'nun bağlı olduğu portu yazın
BAUD_RATE = 9600

# --- MODEL İNDİRİCİ ---
FACE_MODEL_PATH = 'blaze_face_short_range.tflite'
HAND_MODEL_PATH = 'hand_landmarker.task'

def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"[{filename}] bulunamadı. İnternetten indiriliyor...")
        urllib.request.urlretrieve(url, filename)
        print(f"[{filename}] başarıyla indirildi!")

download_model("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", FACE_MODEL_PATH)
download_model("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", HAND_MODEL_PATH)

# --- SERİ HABERLEŞME KURULUMU ---
if ENABLE_SERIAL:
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Arduino'ya bağlanıldı.")
    except Exception as e:
        print(f"Seri port hatası: {e}")
        ENABLE_SERIAL = False

# --- MEDIAPIPE TASKS API KURULUMU ---
face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceDetectorOptions(base_options=face_base_options)
face_detector = vision.FaceDetector.create_from_options(face_options)

hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# ==========================================
# --- SİSTEM DURUMLARI (STATE MACHINE) ---
# ==========================================
STATE_IDLE = 0          # Kilitli, sadece yüz arıyor
STATE_AUTH = 1          # Yüz bulundu, el hareketlerini (şifre) bekliyor
STATE_UNLOCKED = 2      # Şifre doğru, kilit açıldı

current_state = STATE_IDLE
TARGET_SEQUENCE = ["Fist", "Peace", "Open"]
current_sequence = []

# --- ZAMANLAMA VE DEBOUNCE DEĞİŞKENLERİ ---
last_gesture_time = 0
sequence_timeout = 5.0      # Hareketsizlik durumunda başa dön (Saniye)
gesture_cooldown = 1.5      # Aynı hareketi peş peşe okumayı engelle (Saniye)
unlocked_duration = 5.0     # Kilidin açık kalacağı süre (Saniye)
unlocked_start_time = 0

REQUIRED_CONSECUTIVE_FRAMES = 10  # Hareketi onaylamak için gereken ardışık kare sayısı
current_gesture_frames = 0
candidate_gesture = None

def get_gesture(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Başparmak
    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Diğer Parmaklar
    for i in range(1, 5):
        if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Jestleri Sınıflandır
    if fingers == [0, 0, 0, 0, 0] or fingers == [1, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 1, 1, 1, 1] or fingers[1:] == [1, 1, 1, 1]:
        return "Open"

    return "Unknown"

# --- ANA DÖNGÜ ---
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    h, w, c = img.shape

    # 1. YÜZ ALGILAMA (Her zaman çalışır)
    face_result = face_detector.detect(mp_image)
    if len(face_result.detections) > 0:
        # Yüz bulunduğunda, IDLE durumundaysak AUTH durumuna geç
        if current_state == STATE_IDLE:
            current_state = STATE_AUTH
            print("Yüz Algılandı. Şifre Bekleniyor...")

        for detection in face_result.detections:
            bbox = detection.bounding_box
            cv2.rectangle(img, (bbox.origin_x, bbox.origin_y),
                          (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 255, 0), 2)
    else:
        # Yüz kaybolduğunda ve kilit açık değilse güvenliği sağla ve IDLE'a dön
        if current_state == STATE_AUTH:
            print("Yüz kayboldu! Sistem kilitlendi.")
            current_state = STATE_IDLE
            current_sequence = []
            current_gesture_frames = 0

    # ==========================================
    # --- DURUM MAKİNESİ (STATE HANDLING) ---
    # ==========================================

    if current_state == STATE_IDLE:
        cv2.putText(img, "STATE: IDLE (Looking for Face)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "SYSTEM LOCKED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    elif current_state == STATE_AUTH:
        cv2.putText(img, "STATE: AUTH (Enter Passcode)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        hand_result = hand_landmarker.detect(mp_image)
        if len(hand_result.hand_landmarks) > 0:
            landmarks = hand_result.hand_landmarks[0]
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, (0, 255, 255), cv2.FILLED)

            detected_gesture = get_gesture(landmarks)

            # --- DEBOUNCE MANTIĞI (False-Positive Engelleme) ---
            if detected_gesture != "Unknown":
                if detected_gesture == candidate_gesture:
                    current_gesture_frames += 1
                else:
                    candidate_gesture = detected_gesture
                    current_gesture_frames = 1

                # Eğer hareket 10 kare boyunca aynıysa onayla
                if current_gesture_frames >= REQUIRED_CONSECUTIVE_FRAMES:
                    current_time = time.time()

                    # Cooldown süresi kontrolü
                    if (current_time - last_gesture_time) > gesture_cooldown:
                        expected_gesture = TARGET_SEQUENCE[len(current_sequence)]

                        if candidate_gesture == expected_gesture:
                            current_sequence.append(candidate_gesture)
                            last_gesture_time = current_time
                            print(f"Adım Başarılı: {candidate_gesture}. Mevcut Durum: {current_sequence}")

                            # ŞİFRE DOĞRU GİRİLDİYSE UNLOCKED DURUMUNA GEÇ
                            if len(current_sequence) == len(TARGET_SEQUENCE):
                                current_state = STATE_UNLOCKED
                                unlocked_start_time = time.time()
                                print("ACCESS GRANTED! Kilit Açıldı.")
                                if ENABLE_SERIAL:
                                    arduino.write(b'UNLOCK\n')
                        else:
                            if candidate_gesture != current_sequence[-1] if current_sequence else True:
                                print(f"Yanlış Hareket ({candidate_gesture}). Şifre Sıfırlandı.")
                                current_sequence = []
                                last_gesture_time = current_time

                    # İşlem yapıldıktan sonra frame sayacını sıfırla
                    current_gesture_frames = 0

        # --- GÜVENLİK ZAMAN AŞIMI KONTROLÜ ---
        if len(current_sequence) > 0 and (time.time() - last_gesture_time) > sequence_timeout:
            print("Zaman Aşımı! Şifre sıfırlandı.")
            current_sequence = []

        # Ekran Bilgilerini Yazdır
        seq_text = " -> ".join(current_sequence)
        cv2.putText(img, f"Passcode: {seq_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if candidate_gesture and current_gesture_frames > 0:
            cv2.putText(img, f"Reading: {candidate_gesture} ({current_gesture_frames}/{REQUIRED_CONSECUTIVE_FRAMES})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    elif current_state == STATE_UNLOCKED:
        cv2.putText(img, "STATE: UNLOCKED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "ACCESS GRANTED!", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        # Süre dolduğunda sistemi otomatik olarak tekrar kilitle
        time_left = unlocked_duration - (time.time() - unlocked_start_time)
        cv2.putText(img, f"Locking in: {int(time_left)+1}s", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

        if (time.time() - unlocked_start_time) > unlocked_duration:
            print("Süre doldu, sistem tekrar kilitlendi.")
            current_state = STATE_IDLE
            current_sequence = []
            if ENABLE_SERIAL:
                arduino.write(b'LOCK\n') # Arduino'ya tekrar kilitlenmesi için komut gönderilebilir

    cv2.imshow("Air-Pass Security", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ENABLE_SERIAL:
    arduino.close()