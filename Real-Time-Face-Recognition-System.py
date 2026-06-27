import cv2
import face_recognition
import numpy as np 
from pathlib import Path

image_folder = Path("./known_faces")
image_file = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))

if len(image_file) == 0:
    print("Image not found or path is wrong!")
    exit()
    
known_faces = []
known_names = []    

for file in image_file:
    image = cv2.imread(str(file))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_rgb = face_recognition.face_encodings(rgb)
    
    if len(face_rgb) == 0:
        print(f"No face found in {file}")
        continue
    
    known_faces.append(face_rgb[0])
    known_names.append(file.stem)

cap = cv2.VideoCapture(0)
frame_count = 0
face_location = []
face_read = []
names = []

while True:
    
    ret, frame = cap.read()
    if not ret:
        print(" Camera dosn't work! ")
        break
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    if frame_count % 5 == 0:
        face_location = face_recognition.face_locations(rgb_small)
        face_read = face_recognition.face_encodings(rgb_small, face_location)
        names = []
        
        for i in face_read:
            face_distances = face_recognition.face_distance(known_faces, i)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.5:
                name = known_names[best_match_index]
            else:
                name = "Unknown"

            names.append(name)
            
    frame_count += 1
    
    for (top, right, bottom, left), name in zip(face_location, names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        if name == "Unknown":
            color = (0,0,255)

        else:
            color = (0,255,0)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()