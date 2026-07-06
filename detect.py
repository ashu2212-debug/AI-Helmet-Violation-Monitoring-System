from ultralytics import YOLO
import cv2
import os
import time
import csv
from datetime import datetime

# ===========================
# Load trained model
# ===========================
model = YOLO("models/best.pt")

# ===========================
# Violation Folder
# ===========================
VIOLATION_FOLDER = "violations"
os.makedirs(VIOLATION_FOLDER, exist_ok=True)

# ===========================
# CSV File
# ===========================
CSV_FILE = "violations.csv"

# Create CSV if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Class", "Confidence", "Image"])

# ===========================
# Settings
# ===========================
SAVE_COOLDOWN = 5
last_saved_time = 0

# ===========================
# Counters
# ===========================
with_helmet_count = 0
without_helmet_count = 0
violation_count = 0

# ===========================
# Start Webcam
# ===========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam Started...")
print("Press 'Q' to Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Reset frame counters
    with_helmet_count = 0
    without_helmet_count = 0

    # Run YOLO
    results = model(frame, conf=0.4, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Check detections
    for box in results[0].boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == "With Helmet":
            with_helmet_count += 1

        elif class_name == "Without Helmet":
            without_helmet_count += 1

            current_time = time.time()

            if confidence >= 0.70 and (current_time - last_saved_time > SAVE_COOLDOWN):

                # Generate filename
                filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
                filepath = os.path.join(VIOLATION_FOLDER, filename)

                # Save screenshot
                cv2.imwrite(filepath, frame)

                violation_count += 1

                # Save CSV entry
                now = datetime.now()

                with open(CSV_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S"),
                        class_name,
                        f"{confidence:.2f}",
                        filename
                    ])

                print(f"🚨 Violation Saved : {filename}")

                last_saved_time = current_time

    # ===========================
    # Dashboard
    # ===========================
    cv2.putText(
        annotated_frame,
        f"With Helmet : {with_helmet_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Without Helmet : {without_helmet_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        annotated_frame,
        f"Violations Saved : {violation_count}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    # Show webcam
    cv2.imshow("🚨 Helmet Violation Monitoring System", annotated_frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()