from ultralytics import YOLOv10

model = YOLOv10("yolov10m.pt")  

model.train(
    data="dataset.yaml",     
    epochs=300,
    batch=32,
    imgsz=640,
    device="cuda",
    name="train10m_ayuraj"
)