import torch
from torchvision import transforms, models
from PIL import Image
import os

# -----------------------------
# 1️. Obtener la carpeta del script
# -----------------------------
carpeta_script = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# 2️. Cargar modelo entrenado
# -----------------------------
ruta_modelo = os.path.join(carpeta_script, "mi_modelo.pth")
checkpoint = torch.load(ruta_modelo)
class_names = checkpoint['class_names']

model = models.resnet18(weights=None)  # no preentrenado
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # modo evaluación

# -----------------------------
# 3️. Transformaciones de la imagen
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# 4️. Cargar imagen de prueba
# -----------------------------
# Usa ruta absoluta basada en el script
ruta_imagen = os.path.join(carpeta_script, "mi_foto_test.jpg")

if not os.path.exists(ruta_imagen):
    raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

img = Image.open(ruta_imagen)
img = transform(img).unsqueeze(0)  # añadir dimensión batch

# -----------------------------
# 5️. Predecir
# -----------------------------
with torch.no_grad():
    output = model(img)
    _, pred = torch.max(output, 1)

print("Objeto reconocido:", class_names[pred.item()])