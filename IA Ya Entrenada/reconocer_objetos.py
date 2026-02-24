import torch 
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------------------------------
# EXPLICACIÓN librerías usadas
# torch → usar la IA
# torchvision → cargar modelos preentrenados
# PIL → abrir imágenes
# matplotlib → mostrar imágenes
# os → manejar rutas de archivos
# ---------------------------------

# 0. Obtener ruta del script
carpeta_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(carpeta_script, "mi_foto.jpg")

print("Buscando imagen en:", ruta_imagen)

# 1. Cargar modelo preentrenado
# Esto carga una IA llamada: ResNet18 .Es una red neuronal que:
# - ya ha sido entrenada
# - ha visto millones de imágenes
# - puede reconocer 1000 objetos
model = models.resnet18(pretrained=True)
model.eval()

# 2. Transformaciones
# Esto adapta la imagen al formato que la IA necesita.Porque la IA solo acepta imágenes:
# - tamaño específico
# - formato específico
# - valores normalizados
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
# Esto: abre la imagen. La convierte al formato que la IA entiende
# unsqueeze(0) añade una dimensión extra que la IA necesita.
imagen = Image.open(ruta_imagen)
img_tensor = preprocess(imagen).unsqueeze(0)

# 4. Predicción
# Esto es donde ocurre la "magia". La IA devuelve probabilidades como:
# - perro → 85%
# - gato → 10%
# - lobo → 5%
# Y elige el mayor: p.e perro
with torch.no_grad():
    outputs = model(img_tensor)
    _, predicted = outputs.max(1)

# 5. Cargar etiquetas
# La IA devuelve un número, por ejemplo:207
# Esto lo convierte en: golden retriever
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