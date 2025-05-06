yaml_content = """
path: sign_language_dataset
train: train/images
val: valid/images
test: test/images

# Classes
nc: 36
names: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
"""
with open("data.yaml", "w") as f:
    f.write(yaml_content)

from  ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Use smaller model for CPU
model.train(
    data="data.yaml",
    epochs=50,  # Reduced for CPU training
    batch=8,    # Smaller batch size
    device="cpu"  # Force CPU usage
)