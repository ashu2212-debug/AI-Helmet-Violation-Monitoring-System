from ultralytics import YOLO

# Load trained model
model = YOLO("models/best.pt")

# Predict on image
results = model.predict(
    source="images/test.jpg",
    conf=0.4,
    save=True
)

print("Prediction completed successfully!")