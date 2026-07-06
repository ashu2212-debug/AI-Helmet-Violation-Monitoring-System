from ultralytics import YOLO
import cv2
import os

# Load trained model
model = YOLO("models/best.pt")

# Input video
video_path = "videos/test.mp4"

# Output folder
os.makedirs("output", exist_ok=True)
output_path = "output/output_video.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("🎥 Processing video...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame, conf=0.10, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    # Show live preview
    cv2.imshow("Helmet Detection - Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video processing completed.")
print("📁 Saved to:", output_path)