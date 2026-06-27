import cv2
import os
import glob

def extract_and_preprocess_frames(video_path, output_dir, label, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    video_name = os.path.basename(video_path).split('.')[0]
    
    # We will load Haar cascade for simple face detection if dlib fails or is too slow for dummy data
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # We process every 5th frame to save space/time during testing
        if frame_count % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = frame[y:y+h, x:x+w]
            else:
                # If no face detected, take center crop
                h, w, _ = frame.shape
                face = frame[h//2-100:h//2+100, w//2-100:w//2+100]
                
            if face.size == 0:
                continue
                
            # Resize
            face_resized = cv2.resize(face, size)
            
            # Save frame
            out_filename = os.path.join(output_dir, f"{video_name}_frame_{frame_count}.jpg")
            cv2.imwrite(out_filename, face_resized)
            
        frame_count += 1
        
    cap.release()

def process_all_videos():
    real_videos = glob.glob('data/raw_videos/REAL/*.mp4')
    fake_videos = glob.glob('data/raw_videos/FAKE/*.mp4')
    
    print("Processing REAL videos...")
    for video in real_videos:
        extract_and_preprocess_frames(video, 'data/processed_frames/REAL', 'REAL')
        
    print("Processing FAKE videos...")
    for video in fake_videos:
        extract_and_preprocess_frames(video, 'data/processed_frames/FAKE', 'FAKE')

if __name__ == '__main__':
    process_all_videos()
