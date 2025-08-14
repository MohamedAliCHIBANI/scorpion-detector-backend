from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, json
from ultralytics import YOLO

app = FastAPI()

# Autoriser GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://MohamedAliCHIBANI.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = YOLO("best.pt")  # chemin vers ton modèle YOLOv8

@app.get("/")
def home():
    return {"status": "ok", "message": "Server running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            results = model(frame)
            detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": model.names[cls_id]
                    })

            await websocket.send_text(json.dumps({"detections": detections}))

        except Exception as e:
            print("WebSocket error:", e)
            break

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
