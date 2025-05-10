from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    batch=8,
    device="cpu",
    hsv_h=0.015,  # Hue variation
    hsv_s=0.7,    # Saturation variation
    hsv_v=0.4,    # Value variation
    degrees=15,   # Rotation
    translate=0.1, # Translation
    scale=0.5,    # Scale
    fliplr=0.5,   # Horizontal flip
    lr0=0.01,     # Initial learning rate
    lrf=0.1,      # Final learning rate
    warmup_epochs=3,
    mosaic=0.5,   # Mosaic augmentation probability
    mixup=0.1,    # Mixup augmentation probability
    close_mosaic=10,  
    val=True,    
    plots=True    
)