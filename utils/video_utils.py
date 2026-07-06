from ultralytics import YOLO
import cv2
import tempfile
import os

# Load model only once
model = YOLO("models/best.pt")


def detect_video(video):

    # Get actual video path from Gradio
    if isinstance(video, dict):
        video_path = video.get("path")
    else:
        video_path = video

    print("Video Path:", video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    output_path = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.15,
            verbose=False
        )

        annotated = results[0].plot()

        out.write(annotated)

    cap.release()
    out.release()

    print("Saved:", output_path)

    return output_path