import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Ruta al dataset generado (modificada)
dataset_dir = r'C:\Users\ralex\Desktop\proyecto\geometric shapes dataset'
material_dataset_dir = r'C:\Users\ralex\Desktop\proyecto\material'

# Preparar los datos para el modelo de figuras
images = []
labels = []

# Cargar las imágenes y etiquetas desde las subcarpetas
for shape in ['Circle', 'Square', 'Triangle']:
    shape_dir = os.path.join(dataset_dir, shape)
    print(f"Buscando en la ruta: {shape_dir}")
    try:
        for filename in os.listdir(shape_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(shape_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(shape)
    except FileNotFoundError:
        print(f"Error: La carpeta {shape_dir} no se encuentra. Verifica la ruta.")

# Convertir a arrays de numpy
images = np.array(images)
labels = np.array(labels)

# Convertir las etiquetas a números
unique_labels, labels_indices = np.unique(labels, return_inverse=True)

# Normalizar las imágenes
images = images.astype('float32') / 255.0

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels_indices, test_size=0.2, random_state=42)

# Convertir las etiquetas a formato categórico
y_train = to_categorical(y_train, num_classes=len(unique_labels))
y_test = to_categorical(y_test, num_classes=len(unique_labels))

# Definir el nombre del modelo
model_filename = 'geometric_shapes_model.h5'

# Verificar si el modelo ya existe
if os.path.exists(model_filename):
    # Cargar el modelo existente
    model = load_model(model_filename)
    print("Modelo de figuras cargado.")
else:
    # Definir el modelo CNN para figuras
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(unique_labels), activation='softmax')
    ])

    # Compilar el modelo de figuras
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo de figuras
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Guardar el modelo de figuras
    model.save(model_filename)
    print("Modelo de figuras entrenado y guardado.")

# Preparar los datos para el modelo de material
material_images = []
material_labels = []

for material in ['madera', 'acero']:
    material_dir = os.path.join(material_dataset_dir, material)
    print(f"Buscando en la ruta de materiales: {material_dir}")
    try:
        for filename in os.listdir(material_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(material_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (64, 64))
                material_images.append(img)
                material_labels.append(material)
    except FileNotFoundError:
        print(f"Error: La carpeta {material_dir} no se encuentra. Verifica la ruta.")

# Convertir a arrays de numpy
material_images = np.array(material_images)
material_labels = np.array(material_labels)

# Convertir etiquetas a números
unique_materials, material_indices = np.unique(material_labels, return_inverse=True)

# Normalizar las imágenes
material_images = material_images.astype('float32') / 255.0

# Dividir en conjunto de entrenamiento y prueba
X_train_mat, X_test_mat, y_train_mat, y_test_mat = train_test_split(material_images, material_indices, test_size=0.2, random_state=42)

# Convertir las etiquetas a formato categórico
y_train_mat = to_categorical(y_train_mat, num_classes=len(unique_materials))
y_test_mat = to_categorical(y_test_mat, num_classes=len(unique_materials))

# Definir el nombre del modelo de material
material_model_filename = 'material_model.h5'

# Verificar si el modelo de material ya existe
if os.path.exists(material_model_filename):
    # Cargar el modelo existente
    material_model = load_model(material_model_filename)
    print("Modelo de materiales cargado.")
else:
    # Definir el modelo CNN para materiales
    material_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(unique_materials), activation='softmax')
    ])

    # Compilar el modelo de materiales
    material_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo de materiales
    material_model.fit(X_train_mat, y_train_mat, epochs=10, batch_size=32, validation_data=(X_test_mat, y_test_mat))

    # Guardar el modelo de materiales
    material_model.save(material_model_filename)
    print("Modelo de materiales entrenado y guardado.")

# Función para predecir las figuras de una imagen
def predict_shapes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_indices = np.argmax(prediction, axis=1)
    detected_shapes = [unique_labels[i] for i in predicted_indices]
    return detected_shapes

# Función para predecir el material de una imagen
def predict_material(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = material_model.predict(img)
    predicted_index = np.argmax(prediction, axis=1)[0]
    return unique_materials[predicted_index]

# Función para determinar la dificultad según la cantidad de figuras
def get_difficulty(num_shapes):
    if num_shapes == 1:
        return 'Baja'
    elif num_shapes == 2:
        return 'Media'
    else:
        return 'Alta'

def save_to_pdf(detected_shapes, material, difficulty, shape_image_path, material_image_path):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{difficulty}_{material}_{date_str}.pdf"

    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter

    # Título en la parte superior
    c.drawString(100, height - 100, f"Dificultad: {difficulty}")
    c.drawString(100, height - 120, f"Material: {material}")

    # Mostrar imagen de la figura
    c.drawImage(shape_image_path, 100, height - 300, width=200, height=200)

    # Mostrar imagen del material (predicción) al lado
    c.drawImage(material_image_path, 350, height - 300, width=200, height=200)

    # Mostrar nombres de las figuras debajo de las imágenes
    shapes_text = ', '.join(detected_shapes)
    c.drawString(100, height - 520, f"Figuras: {shapes_text}")

    c.save()
    print(f"PDF guardado como {pdf_filename}")

# Función para mostrar la interfaz gráfica con ambas imágenes
def show_result(detected_shapes, shape_image_path, material, material_image_path):
    num_shapes = len(detected_shapes)
    difficulty = get_difficulty(num_shapes)

    root = tk.Tk()
    root.title(f"Dificultad: {difficulty} - {material}")

    # Mostrar dificultad y material
    difficulty_label = Label(root, text=f"Dificultad: {difficulty} - Material: {material}", font=("Arial", 16))
    difficulty_label.pack(pady=10)

    # Mostrar imagen de la figura
    shape_img = Image.open(shape_image_path)
    shape_img = shape_img.resize((200, 200))
    shape_img_tk = ImageTk.PhotoImage(shape_img)
    shape_label = Label(root, image=shape_img_tk)
    shape_label.image = shape_img_tk
    shape_label.pack(side="left", padx=10, pady=10)

    # Mostrar nombres de las figuras
    shapes_text = ', '.join(detected_shapes)
    name_label = Label(root, text=f"Figuras: {shapes_text}", font=("Arial", 16))
    name_label.pack(pady=(30, 10), padx=10, expand=True, fill='x')  # Mover más hacia abajo y centrar

    # Crear un contenedor para el botón en la parte inferior derecha
    button_frame = tk.Frame(root)
    button_frame.pack(side="bottom", anchor="se", padx=10, pady=10)

# Botón para generar PDF ubicado en la esquina inferior derecha
    pdf_button = Button(button_frame, text="Generar PDF", command=lambda: save_to_pdf(detected_shapes, material, difficulty, shape_image_path, material_image_path))
    pdf_button.pack()


    # Ejecutar la interfaz
    root.mainloop()

#imagen a usar
if __name__ == "__main__":
    shape_image_path = r'C:\Users\ralex\Desktop\proyecto\ima\cuadrado.png'  # Imagen de figura
    material_image_path = r'C:\Users\ralex\Desktop\proyecto\ima\madera-de-parota-1.jpg'  # Imagen de material
    
    detected_shapes = predict_shapes(shape_image_path)
    material = predict_material(material_image_path)
    
    if detected_shapes is not None and material is not None:
        show_result(detected_shapes, shape_image_path, material, material_image_path)
