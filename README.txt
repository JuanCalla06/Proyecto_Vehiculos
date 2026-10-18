REQUISITOS DEL PROYECTO

Python: 3.12.x

Dependencias:
TensorFlow: 2.21.0
OpenCV
NumPy
CustomTkinter
Pillow

EJECUCIÓN:

1. Crear entorno virtual
2. Activarlo
3. Instalar requirements.txt
4. Ejecutar:

python main.py

# Sistema de Control Vehicular

Sistema de clasificación y control de acceso vehicular desarrollado en Python utilizando TensorFlow, Keras, OpenCV y CustomTkinter.

## 1. Requisitos

- Python 3.12.x
- Cámara web
- Windows
- VS Code (opcional)

## 2. Estructura del proyecto

Proyecto Vehiculos/
│
├── main.py
├── keras_model.h5
├── labels.txt
├── requirements.txt
├── README.md
│
├── permisos_vehiculos.json
└── historial_vehicular.csv




Si se tiene problemas con la cámara, modificar la variable índice de cámara de 1(por defecto) a 0 (trabajará con la webcam)