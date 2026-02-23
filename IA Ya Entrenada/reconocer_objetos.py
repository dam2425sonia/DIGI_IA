import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------------------------------
# SOLUCIÓN: obtener ruta del script
# ---------------------------------
carpeta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(carpeta_script, "mi_foto1.jpg")

print("Buscando imagen en:", ruta_imagen)

# 1. Cargar modelo preentrenado
model = models.resnet18(pretrained=True)
model.eval()

# 2. Transformaciones
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Cargar imagen
imagen = Image.open(ruta_imagen)
img_tensor = preprocess(imagen).unsqueeze(0)

# 4. Predicción
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = outputs.max(1)

# 5. Cargar etiquetas
import json, urllib.request

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(urllib.request.urlopen(url).read())

resultado = labels[predicted.item()]
print("Objeto reconocido:", resultado)

# 6. Mostrar imagen
plt.imshow(imagen)
plt.title(resultado)
plt.axis('off')
plt.show()