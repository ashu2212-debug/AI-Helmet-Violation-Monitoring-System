from ultralytics import YOLO
import cv2
import tempfile

# Load model only once
model = YOLO("models/best.pt")


def detect_image(image):
    """
    Takes a PIL image from Gradio,
    runs YOLO detection,
    returns annotated image.
    """

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        temp_path = temp.name

    # Run prediction
    results = model.predict(temp_path, conf=0.15, verbose=False)

    # Draw detections
    annotated = results[0].plot()

    # Convert BGR → RGB
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated