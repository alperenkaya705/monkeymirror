import cv2
import mediapipe as mp
import os
import math
import numpy as np


GORSEL_KLASORU = "Gorseller"
KAMERA_ID = 0
PENCERE_ADI = "Monkey AI"
HEDEF_YUKSEKLIK = 500


mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    refine_landmarks=False
)


images = {}
file_names = {
    "normal": "monkey_normal.png",
    "yes": "monkey_yes.png",
    "ok": "monkey_ok.png",
    "idea": "monkey_idea.png",
    "thinking": "monkey_thinking.png",
    "shocked": "monkey_shocked.png",
    "dontsee": "monkey_dontsee.png",
    "donthear": "monkey_donthear.png",
    "dontspeak": "monkey_dontspeak.png"
}

for key, filename in file_names.items():
    path = os.path.join(GORSEL_KLASORU, filename)
    img = cv2.imread(path)
    if img is None:
        images[key] = np.zeros((HEDEF_YUKSEKLIK, HEDEF_YUKSEKLIK, 3), dtype=np.uint8)
    else:
        images[key] = cv2.resize(img, (HEDEF_YUKSEKLIK, HEDEF_YUKSEKLIK))


def find_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_fingers(lm_list):
    fingers = []
    # Başparmak
    if lm_list[4][2] < lm_list[3][2]: fingers.append(1) 
    else: fingers.append(0)
    
    # Diğer 4 parmak
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if lm_list[tip][2] < lm_list[pip][2]: fingers.append(1)
        else: fingers.append(0)
    return fingers


cap = cv2.VideoCapture(KAMERA_ID)

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb_frame)
    results_face = face_mesh.process(rgb_frame)

    current_status = "normal"
    
    #YÜZ ANALİZİ
    face_points = {}
    is_shocked = False
    mouth_center = (0, 0)
    
    if results_face.multi_face_landmarks:
        fl = results_face.multi_face_landmarks[0]
        # Kritik noktalar
        indices = [1, 10, 152, 33, 263, 13, 14] 
        for idx in indices:
            lm = fl.landmark[idx]
            face_points[idx] = (int(lm.x * w), int(lm.y * h))
        
        mouth_center = ((face_points[13][0] + face_points[14][0]) // 2, 
                        (face_points[13][1] + face_points[14][1]) // 2)

        # Şok hesaplama için ağız açıklığı
        mouth_dist = find_distance(face_points[13], face_points[14])
        face_height = find_distance(face_points[10], face_points[152])
        if mouth_dist > face_height * 0.08: # %8 açıklık 
            is_shocked = True

    #EL ANALİZİ
    if results_hands.multi_hand_landmarks and face_points:
        hands_data = []
        for hlm in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
            lm_list = [(id, int(lm.x * w), int(lm.y * h)) for id, lm in enumerate(hlm.landmark)]
            
            hands_data.append({
                "center": (lm_list[9][1], lm_list[9][2]),
                "fingers": get_fingers(lm_list),
                "index_tip": (lm_list[8][1], lm_list[8][2]),
                "thumb_tip": (lm_list[4][1], lm_list[4][2]),
                "thumb_ip":  (lm_list[3][1], lm_list[3][2])
            })

        num_hands = len(hands_data)
        override_gesture = False 

        # ÇİFT EL HAREKETLERİ
        if num_hands == 2:
            h1 = hands_data[0]["center"]
            h2 = hands_data[1]["center"]
            
            # konuşmuyor
            if find_distance(h1, mouth_center) < 90 and \
               find_distance(h2, mouth_center) < 90 and \
               find_distance(h1, h2) < 130:
                current_status = "dontspeak"
                override_gesture = True

            # görmüyor
            if not override_gesture:
                eye_left, eye_right = face_points[33], face_points[263]
                dist_h1_eyes = min(find_distance(h1, eye_left), find_distance(h1, eye_right))
                dist_h2_eyes = min(find_distance(h2, eye_left), find_distance(h2, eye_right))
                
                if dist_h1_eyes < 80 and dist_h2_eyes < 80:
                    current_status = "dontsee"
                    override_gesture = True

            # duymuyor
            if not override_gesture:
                if find_distance(h1, h2) > 150: 
                    avg_y = (h1[1] + h2[1]) / 2
                    if face_points[33][1] - 50 < avg_y < face_points[152][1]: 
                        
                        if (h1[0] < face_points[1][0] < h2[0]) or (h2[0] < face_points[1][0] < h1[0]):
                             current_status = "donthear"
                             override_gesture = True

        # düşünme
        if not override_gesture:
            for hand in hands_data:
                # İşaret parmağı çenede
                if find_distance(hand["index_tip"], face_points[152]) < 70 and hand["fingers"][1] == 1:
                    current_status = "thinking"
                    override_gesture = True
                    break

        # OK YES IDEA SHOCKED
        if not override_gesture:
            # ŞOK 
            hands_near_mouth = False
            for hand in hands_data:
                if find_distance(hand["center"], mouth_center) < 100:
                    hands_near_mouth = True
            
            if is_shocked and not hands_near_mouth:
                current_status = "shocked"
            
            else:
                
                for hand in hands_data:
                    
                    if find_distance(hand["center"], mouth_center) > 120:
                        f = hand["fingers"]
                        
                        # OK 
                        dist_ok = find_distance(hand["thumb_tip"], hand["index_tip"])
                        if dist_ok < 70 and (f[2] == 1 or f[3] == 1):
                            current_status = "ok"
                            break
                        
                        #IDEA
                        if f[1]==1 and f[2]==0 and f[3]==0 and f[4]==0:
                            current_status = "idea"
                            break
                        
                        #YES
                        if f[1]==0 and f[2]==0 and f[3]==0 and f[4]==0:
                            if hand["thumb_tip"][1] < hand["thumb_ip"][1] - 10:
                                current_status = "yes"
                                break

    # Sadece Yüz var, El Yoksa -> Şok Kontrolü
    elif face_points and not results_hands.multi_hand_landmarks:
        if is_shocked:
            current_status = "shocked"

    # Ekran komutları
    monkey_img = images.get(current_status, images["normal"])
    
    aspect_ratio = w / h
    new_w = int(HEDEF_YUKSEKLIK * aspect_ratio)
    frame_resized = cv2.resize(frame, (new_w, HEDEF_YUKSEKLIK))
    
    final_img = np.zeros((HEDEF_YUKSEKLIK, new_w + HEDEF_YUKSEKLIK, 3), dtype=np.uint8)
    final_img[:, :new_w] = frame_resized
    final_img[:, new_w:] = monkey_img


        # "q " tuşuna basarak kapatılr
    cv2.imshow(PENCERE_ADI, final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()