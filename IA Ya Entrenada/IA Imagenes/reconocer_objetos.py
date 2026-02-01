import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1️⃣ Cargar modelo preentrenado
model = models.resnet18(pretrained=True)
model.eval()  # poner en modo evaluación

# 2️⃣ Transformaciones de la imagen
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 3️⃣ Cargar la imagen
imagen = Image.open("mi_foto.jpg")  # pon aquí tu foto
img_tensor = preprocess(imagen).unsqueeze(0)  # añadir dimensión batch

# 4️⃣ Hacer predicción
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = outputs.max(1)

# 5️⃣ Traducir número a etiqueta
# Descargar etiquetas de ImageNet
import json, urllib.request
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(urllib.request.urlopen(url).read())
print("Objeto reconocido:", labels[predicted.item()])

# 6️⃣ Mostrar la imagen
plt.imshow(imagen)
plt.title(labels[predicted.item()])
plt.axis('off')
plt.show()
