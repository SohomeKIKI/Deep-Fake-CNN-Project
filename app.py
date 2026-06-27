from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import cv2
import numpy as np
import os
import shutil

app = FastAPI(title="Deepfake Detection API")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = "saved_models/baseline_cnn.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Fallback Face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame, img_size=(224, 224)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]
    else:
        # Fallback to center crop
        h, w, _ = frame.shape
        face = frame[h//2-100:h//2+100, w//2-100:w//2+100]
        if face.size == 0:
            face = frame
            
    face_resized = cv2.resize(face, img_size)
    
    # Simulate JPEG compression to match the training data saved via imwrite
    _, buffer = cv2.imencode('.jpg', face_resized)
    face_jpeg = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
    face_normalized = np.array(face_jpeg, dtype=np.float32) / 255.0
    return face_normalized

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
        
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, file.filename)
    
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    predictions = []
    
    # Check if image or video
    ext = file.filename.split('.')[-1].lower()
    if ext in ['jpg', 'jpeg', 'png']:
        img = cv2.imread(temp_file)
        if img is not None:
            processed_img = process_frame(img)
            pred = model.predict(np.expand_dims(processed_img, axis=0))[0][0]
            predictions.append(pred)
    elif ext in ['mp4', 'avi', 'mov']:
        cap = cv2.VideoCapture(temp_file)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Process a few frames to keep it fast
            if frame_count % 15 == 0:
                processed_img = process_frame(frame)
                pred = model.predict(np.expand_dims(processed_img, axis=0))[0][0]
                print(f"Frame {frame_count} prediction: {pred}")
                predictions.append(pred)
            frame_count += 1
            if len(predictions) >= 5: # max 5 frames for quick response
                break
        cap.release()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    # Cleanup
    try:
        os.remove(temp_file)
    except:
        pass
        
    if len(predictions) == 0:
        raise HTTPException(status_code=400, detail="Could not process media.")
        
    avg_pred = float(np.mean(predictions))
    
    # 0 = FAKE, 1 = REAL
    # Usually we want the confidence of the predicted class
    if avg_pred >= 0.5:
        result = "REAL"
        confidence = avg_pred * 100
    else:
        result = "FAKE"
        confidence = (1 - avg_pred) * 100
        
    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
