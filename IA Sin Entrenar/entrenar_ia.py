import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Transformaciones
    #Resize((224,224)) → todas las imágenes se redimensionan a 224x224 píxeles porque ResNet18 requiere este tamaño de entrada.
    #ToTensor() → convierte la imagen en un tensor que PyTorch puede procesar (valores entre 0 y 1).
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Cargar dataset
    #ImageFolder busca todas las subcarpetas en dataset/ y asigna cada subcarpeta como una clase.
    #Ejemplo: 'manzana' → 0, 'platano' → 1, 'perro' → 2, 'gato' → 3
    #DataLoader sirve para cargar los datos en lotes (batch_size=4) y barajar las imágenes (shuffle=True) para que la IA no aprenda en orden fijo.
dataset = datasets.ImageFolder("dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Guardar nombres de clases
    #Esto guarda los nombres de tus clases en una lista: ['manzana', 'platano', 'perro', 'gato'].
    #Se usa después para decodificar la predicción numérica a un nombre legible.
class_names = dataset.classes
print("Clases detectadas:", class_names)  # ['manzana', 'platano', 'perro', 'gato']

# Modelo preentrenado
    #ResNet18 es una red neuronal convolucional profunda, ya entrenada en millones de imágenes de ImageNet.
    #model.fc → la última capa de ResNet18 que produce predicciones.
    #Por defecto está entrenada para 1000 clases de ImageNet.
    #La reemplazamos con nn.Linear(num_ftrs, len(class_names)) para que tenga 4 salidas, una para cada clase de tu dataset.
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # 4 clases

# Configurar pérdida y optimizador
    #CrossEntropyLoss → función de pérdida para clasificación múltiple.
    #Adam → optimizador que ajusta los pesos de la red durante el entrenamiento.
    #lr=0.001 → velocidad a la que aprende la IA.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
    #Mueve la red a la GPU si está disponible, sino usa CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Por cada época y cada lote (batch) de imágenes:
    #inputs, labels → imágenes y sus etiquetas.
    #optimizer.zero_grad() → reinicia gradientes.
    #outputs = model(inputs) → la IA predice las clases.
    #loss = criterion(outputs, labels) → calcula qué tan mal está prediciendo.
    #loss.backward() → calcula gradientes.
    #optimizer.step() → actualiza los pesos.
    #Al final de cada época, imprimimos la pérdida promedio (Loss).
epochs = 5  # prueba primero con pocas épocas
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

# Guardar modelo y nombres de clases
    #Guardamos los pesos del modelo y la lista de nombres de clases.
    #Esto permite luego cargar el modelo entrenado sin necesidad de entrenarlo otra vez.
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, "mi_modelo.pth")

print("Entrenamiento terminado y modelo guardado ✅")
