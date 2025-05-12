from ultralytics import YOLOv10

# Load the trained model
model = YOLOv10("yolov10/runs/detect/train10m_ayuraj/weights/best.pt")

# Run predictions on test set
model.predict(
    source="datasets/data/ayuraj_dataset/test/images",
    save=True,
    conf=0.25,
    name="predict_yolov10m_ayuraj"
)