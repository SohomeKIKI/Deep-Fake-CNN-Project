import cv2
import numpy as np
import os

def create_dummy_video(filename, is_real=True):
    # 8 seconds video at 30 fps -> 240 frames
    fps = 30
    duration = 8
    num_frames = fps * duration
    width, height = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Create some moving pattern
        if is_real:
            color = (0, 255, 0) # Green for real
        else:
            color = (0, 0, 255) # Red for fake
            
        x = int(width / 2 + 100 * np.sin(i * 0.1))
        y = int(height / 2 + 100 * np.cos(i * 0.1))
        
        cv2.circle(frame, (x, y), 50, color, -1)
        # Add a mock "face" rectangle to ensure dlib/opencv can process something if we use haar cascades
        cv2.rectangle(frame, (x-40, y-40), (x+40, y+40), (255, 255, 255), 2)
        out.write(frame)
        
    out.release()

def generate_data():
    os.makedirs('data/raw_videos/REAL', exist_ok=True)
    os.makedirs('data/raw_videos/FAKE', exist_ok=True)
    
    print("Generating REAL videos...")
    for i in range(2):
        create_dummy_video(f'data/raw_videos/REAL/real_video_{i}.mp4', is_real=True)
        
    print("Generating FAKE videos...")
    for i in range(2):
        create_dummy_video(f'data/raw_videos/FAKE/fake_video_{i}.mp4', is_real=False)

if __name__ == '__main__':
    generate_data()
