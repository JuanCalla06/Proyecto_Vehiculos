import os
import json
import csv
import threading
import time
from datetime import datetime

# --------------------------------------------------
# KERAS LEGACY: CONFIGURAR ANTES DE TENSORFLOW
# --------------------------------------------------

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk


# ==================================================
# 1. CONFIGURACIÓN GENERAL
# ==================================================

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO = os.path.join(CARPETA_PROYECTO, "keras_model.h5")
RUTA_ETIQUETAS = os.path.join(CARPETA_PROYECTO, "labels.txt")
RUTA_PERMISOS = os.path.join(
    CARPETA_PROYECTO, "permisos_vehiculos.json"
)
RUTA_HISTORIAL = os.path.join(
    CARPETA_PROYECTO, "historial_vehicular.csv"
)

ANCHO_CAMARA = 640
ALTO_CAMARA = 480
INDICE_CAMARA = 1

INTERVALO_IA = 0.15
INTERVALO_UI = 30
PREDICCIONES_ESTABLES = 3

PERMISOS_INICIALES = {
    "WH173": True,
    "BL031": False,
    "RE337": True,
}

NOMBRES_VEHICULOS = list(PERMISOS_INICIALES.keys())


# ==================================================
# 2. CARGAR PERMISOS PERSISTENTES
# ==================================================

def cargar_permisos():
    permisos = PERMISOS_INICIALES.copy()

    if os.path.exists(RUTA_PERMISOS):
        try:
            with open(RUTA_PERMISOS, "r", encoding="utf-8") as archivo:
                datos = json.load(archivo)

            for vehiculo in NOMBRES_VEHICULOS:
                valor = datos.get(vehiculo)

                if isinstance(valor, bool):
                    permisos[vehiculo] = valor

        except (OSError, json.JSONDecodeError) as error:
            print("No se pudieron cargar los permisos:", error)

    return permisos


def guardar_permisos():
    temporal = RUTA_PERMISOS + ".tmp"

    with open(temporal, "w", encoding="utf-8") as archivo:
        json.dump(
            vehiculos_autorizados,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    os.replace(temporal, RUTA_PERMISOS)


vehiculos_autorizados = cargar_permisos()


# ==================================================
# 3. PREPARAR HISTORIAL CSV
# ==================================================

COLUMNAS_CSV = [
    "fecha_hora",
    "vehiculo",
    "confianza_pct",
    "estado_acceso",
    "tipo_evento",
]


def preparar_historial():
    existe = (
        os.path.exists(RUTA_HISTORIAL)
        and os.path.getsize(RUTA_HISTORIAL) > 0
    )

    if not existe:
        with open(
            RUTA_HISTORIAL,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(COLUMNAS_CSV)


def registrar_evento(vehiculo, confianza, acceso, tipo_evento):
    try:
        with open(
            RUTA_HISTORIAL,
            "a",
            newline="",
            encoding="utf-8-sig"
        ) as archivo:
            escritor = csv.writer(archivo)

            escritor.writerow([
                datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                vehiculo,
                round(confianza * 100, 2),
                acceso,
                tipo_evento,
            ])

    except OSError as error:
        print("No se pudo guardar el historial:", error)


preparar_historial()


# ==================================================
# 4. CARGAR MODELO Y ETIQUETAS
# ==================================================

print("Cargando modelo de inteligencia artificial...")

model = tf.keras.models.load_model(RUTA_MODELO)

with open(RUTA_ETIQUETAS, "r", encoding="utf-8") as archivo:
    labels = []

    for linea in archivo:
        partes = linea.strip().split(maxsplit=1)

        if len(partes) == 2:
            labels.append(partes[1])

if not labels:
    raise ValueError("El archivo labels.txt no contiene clases válidas.")

if model.output_shape[-1] != len(labels):
    raise ValueError(
        "La cantidad de salidas del modelo no coincide con labels.txt."
    )

print("Modelo cargado correctamente.")
print("Clases:", labels)


# ==================================================
# 5. ABRIR CÁMARA
# ==================================================

camara = cv2.VideoCapture(INDICE_CAMARA)

if not camara.isOpened():
    camara.release()
    raise RuntimeError(
        f"No se pudo abrir la cámara con índice {INDICE_CAMARA}."
    )

camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)
camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ==================================================
# 6. VARIABLES COMPARTIDAS ENTRE HILOS
# ==================================================

bloqueo = threading.Lock()
detener = threading.Event()

ultimo_frame = None
ultimo_resultado = None
numero_resultado = 0


# ==================================================
# 7. ESTADO DE DETECCIÓN E HISTORIAL
# ==================================================

ultimo_nombre_detectado = None
ultima_probabilidad = 0.0

ultimo_vehiculo_evento = None
ultimo_acceso_evento = None


def es_sin_vehiculo(nombre):
    # Admite las variantes habituales de esta etiqueta.
    return nombre.strip().lower() in (
        "sin vehiculo",
        "sin vehículo",
    )


def calcular_estado_acceso(nombre):
    if es_sin_vehiculo(nombre):
        return "ESPERANDO VEHICULO"

    if vehiculos_autorizados.get(nombre, False):
        return "ACCESO PERMITIDO"

    return "ACCESO DENEGADO"


def procesar_evento(nombre, confianza, motivo):
    """
    Registra únicamente una detección inicial o un cambio
    de vehículo/estado de acceso.
    Se llama desde el hilo principal de la interfaz.
    """
    global ultimo_vehiculo_evento, ultimo_acceso_evento

    acceso = calcular_estado_acceso(nombre)

    cambio_vehiculo = nombre != ultimo_vehiculo_evento
    cambio_acceso = acceso != ultimo_acceso_evento

    if not cambio_vehiculo and not cambio_acceso:
        return

    if ultimo_vehiculo_evento is None:
        tipo_evento = "DETECCION_INICIAL"
    elif motivo == "CAMBIO_DE_PERMISO":
        tipo_evento = "CAMBIO_DE_PERMISO"
    else:
        tipo_evento = "CAMBIO_DE_DETECCION"

    registrar_evento(
        vehiculo=nombre,
        confianza=confianza,
        acceso=acceso,
        tipo_evento=tipo_evento
    )

    ultimo_vehiculo_evento = nombre
    ultimo_acceso_evento = acceso


# ==================================================
# 8. CONFIGURAR INTERFAZ
# ==================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ventana = ctk.CTk()
ventana.title("Sistema de Control Vehicular")
ventana.geometry("1200x780")
ventana.minsize(1000, 700)

ventana.grid_columnconfigure(0, weight=3)
ventana.grid_columnconfigure(1, weight=2)
ventana.grid_rowconfigure(1, weight=1)


# ==================================================
# 9. TÍTULO
# ==================================================

titulo = ctk.CTkLabel(
    ventana,
    text="SISTEMA DE CONTROL VEHICULAR",
    font=ctk.CTkFont(size=26, weight="bold")
)

titulo.grid(
    row=0,
    column=0,
    columnspan=2,
    padx=20,
    pady=20
)


# ==================================================
# 10. PANEL DE CÁMARA
# ==================================================

panel_camara = ctk.CTkFrame(
    ventana,
    corner_radius=15
)

panel_camara.grid(
    row=1,
    column=0,
    padx=(20, 10),
    pady=(0, 20),
    sticky="nsew"
)

panel_camara.grid_rowconfigure(0, weight=1)
panel_camara.grid_columnconfigure(0, weight=1)

etiqueta_video = ctk.CTkLabel(
    panel_camara,
    text="Iniciando cámara..."
)

etiqueta_video.grid(
    row=0,
    column=0,
    padx=15,
    pady=15,
    sticky="nsew"
)


# ==================================================
# 11. PANEL DE INFORMACIÓN
# ==================================================

panel_info = ctk.CTkFrame(
    ventana,
    corner_radius=15
)

panel_info.grid(
    row=1,
    column=1,
    padx=(10, 20),
    pady=(0, 20),
    sticky="nsew"
)


# ==================================================
# 12. DETECCIÓN ACTUAL
# ==================================================

texto_deteccion = ctk.CTkLabel(
    panel_info,
    text="DETECCIÓN ACTUAL",
    font=ctk.CTkFont(size=20, weight="bold")
)

texto_deteccion.pack(padx=15, pady=(20, 10))

texto_vehiculo = ctk.CTkLabel(
    panel_info,
    text="Esperando...",
    font=ctk.CTkFont(size=27, weight="bold")
)

texto_vehiculo.pack(padx=15, pady=5)

texto_confianza = ctk.CTkLabel(
    panel_info,
    text="Confianza: --",
    font=ctk.CTkFont(size=17)
)

texto_confianza.pack(padx=15, pady=5)

texto_acceso = ctk.CTkLabel(
    panel_info,
    text="SIN DETECCIÓN",
    font=ctk.CTkFont(size=20, weight="bold"),
    text_color="#eab308"
)

texto_acceso.pack(padx=15, pady=(10, 20))


# ==================================================
# 13. ADMINISTRACIÓN DE PERMISOS
# ==================================================

titulo_permisos = ctk.CTkLabel(
    panel_info,
    text="ADMINISTRAR PERMISOS",
    font=ctk.CTkFont(size=18, weight="bold")
)

titulo_permisos.pack(padx=15, pady=(0, 10))

filas_permisos = {}


def actualizar_fila_permiso(vehiculo):
    autorizado = vehiculos_autorizados[vehiculo]

    etiqueta = filas_permisos[vehiculo]["estado"]
    boton = filas_permisos[vehiculo]["boton"]

    if autorizado:
        etiqueta.configure(
            text="AUTORIZADO",
            text_color="#22c55e"
        )

        boton.configure(
            text="Denegar",
            fg_color="#b91c1c",
            hover_color="#991b1b"
        )

    else:
        etiqueta.configure(
            text="DENEGADO",
            text_color="#ef4444"
        )

        boton.configure(
            text="Autorizar",
            fg_color="#15803d",
            hover_color="#166534"
        )


def cambiar_permiso(vehiculo):
    global ultima_probabilidad

    vehiculos_autorizados[vehiculo] = (
        not vehiculos_autorizados[vehiculo]
    )

    try:
        guardar_permisos()
    except OSError as error:
        print("No se pudieron guardar los permisos:", error)

    actualizar_fila_permiso(vehiculo)

    # Si se modifica el permiso del vehículo que está
    # detectado, actualizar inmediatamente el acceso.
    if ultimo_nombre_detectado is not None:
        actualizar_estado_visual(
            ultimo_nombre_detectado,
            ultima_probabilidad
        )

        procesar_evento(
            ultimo_nombre_detectado,
            ultima_probabilidad,
            "CAMBIO_DE_PERMISO"
        )


for vehiculo in NOMBRES_VEHICULOS:

    fila = ctk.CTkFrame(
        panel_info,
        fg_color="transparent"
    )

    fila.pack(fill="x", padx=15, pady=4)

    nombre = ctk.CTkLabel(
        fila,
        text=vehiculo,
        font=ctk.CTkFont(size=16, weight="bold"),
        width=75
    )

    nombre.pack(side="left", padx=(0, 5))

    estado = ctk.CTkLabel(
        fila,
        text="",
        font=ctk.CTkFont(size=13, weight="bold"),
        width=110
    )

    estado.pack(side="left", padx=5)

    boton = ctk.CTkButton(
        fila,
        text="",
        width=90,
        command=lambda v=vehiculo: cambiar_permiso(v)
    )

    boton.pack(side="right", padx=3)

    filas_permisos[vehiculo] = {
        "estado": estado,
        "boton": boton
    }

    actualizar_fila_permiso(vehiculo)


# ==================================================
# 14. ACTUALIZAR ESTADO VISUAL
# ==================================================

def actualizar_estado_visual(nombre, confianza):
    texto_vehiculo.configure(
        text=(
            "SIN VEHÍCULO"
            if es_sin_vehiculo(nombre)
            else nombre
        )
    )

    texto_confianza.configure(
        text=f"Confianza: {confianza * 100:.2f}%"
    )

    acceso = calcular_estado_acceso(nombre)

    if acceso == "ACCESO PERMITIDO":
        texto_acceso.configure(
            text="✓ ACCESO PERMITIDO",
            text_color="#22c55e"
        )

    elif acceso == "ACCESO DENEGADO":
        texto_acceso.configure(
            text="✗ ACCESO DENEGADO",
            text_color="#ef4444"
        )

    else:
        texto_acceso.configure(
            text="ESPERANDO VEHÍCULO",
            text_color="#eab308"
        )


# ==================================================
# 15. HILO DE CAPTURA DE VIDEO
# ==================================================

def capturar_video():
    global ultimo_frame

    while not detener.is_set():

        ret, frame = camara.read()

        if not ret:
            detener.wait(0.02)
            continue

        with bloqueo:
            ultimo_frame = frame


# ==================================================
# 16. HILO DE INTELIGENCIA ARTIFICIAL
# ==================================================

def ejecutar_ia():
    global ultimo_resultado, numero_resultado

    clase_candidata = None
    repeticiones = 0
    clase_estable = None

    while not detener.is_set():

        inicio = time.monotonic()

        with bloqueo:
            frame = (
                None if ultimo_frame is None
                else ultimo_frame.copy()
            )

        if frame is None:
            detener.wait(0.05)
            continue

        try:
            # PREPROCESAMIENTO ORIGINAL
            imagen_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            imagen_redimensionada = cv2.resize(
                imagen_rgb,
                (224, 224)
            )

            imagen_normalizada = (
                imagen_redimensionada.astype(np.float32)
                / 255.0
            )

            imagen_entrada = np.expand_dims(
                imagen_normalizada,
                axis=0
            )

            # PREDICCIÓN
            prediccion = model.predict(
                imagen_entrada,
                verbose=0
            )

            indice = int(np.argmax(prediccion[0]))
            confianza = float(prediccion[0][indice])
            nombre = labels[indice]

            # ESTABILIZACIÓN POR PREDICCIONES CONSECUTIVAS
            if nombre == clase_candidata:
                repeticiones += 1
            else:
                clase_candidata = nombre
                repeticiones = 1

            if repeticiones >= PREDICCIONES_ESTABLES:
                clase_estable = clase_candidata

                # Solo se publica la clase estable.
                # La interfaz lee este resultado en su hilo.
                with bloqueo:
                    ultimo_resultado = (
                        clase_estable,
                        confianza
                    )
                    numero_resultado += 1

        except Exception as error:
            print("Error durante la inferencia:", error)
            detener.wait(0.2)

        transcurrido = time.monotonic() - inicio
        espera = max(0, INTERVALO_IA - transcurrido)

        detener.wait(espera)


# ==================================================
# 17. ACTUALIZAR INTERFAZ SIN BLOQUEARLA
# ==================================================

ultimo_resultado_mostrado = 0


def actualizar_interfaz():
    global ultimo_resultado_mostrado
    global ultimo_nombre_detectado, ultima_probabilidad

    if detener.is_set():
        return

    # OBTENER EL FOTOGRAMA MÁS RECIENTE
    with bloqueo:
        frame = (
            None if ultimo_frame is None
            else ultimo_frame.copy()
        )

        resultado = ultimo_resultado
        numero = numero_resultado

    # MOSTRAR VIDEO
    if frame is not None:
        imagen_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagen_pil = Image.fromarray(imagen_rgb)
        imagen_tk = ImageTk.PhotoImage(image=imagen_pil)

        etiqueta_video.configure(
            image=imagen_tk,
            text=""
        )

        etiqueta_video.image = imagen_tk

    # PROCESAR SOLO RESULTADOS NUEVOS
    if (
        resultado is not None
        and numero != ultimo_resultado_mostrado
    ):

        nombre, confianza = resultado

        ultimo_nombre_detectado = nombre
        ultima_probabilidad = confianza

        actualizar_estado_visual(nombre, confianza)

        procesar_evento(
            nombre,
            confianza,
            "CAMBIO_DE_DETECCION"
        )

        ultimo_resultado_mostrado = numero

    ventana.after(
        INTERVALO_UI,
        actualizar_interfaz
    )


# ==================================================
# 18. CIERRE SEGURO
# ==================================================

def cerrar_programa():
    if detener.is_set():
        return

    detener.set()

    try:
        guardar_permisos()
    except OSError as error:
        print("No se pudieron guardar los permisos:", error)

    camara.release()
    ventana.destroy()


ventana.protocol(
    "WM_DELETE_WINDOW",
    cerrar_programa
)


# ==================================================
# 19. INICIAR SISTEMA
# ==================================================

hilo_camara = threading.Thread(
    target=capturar_video,
    name="HiloCamara",
    daemon=True
)

hilo_ia = threading.Thread(
    target=ejecutar_ia,
    name="HiloIA",
    daemon=True
)

hilo_camara.start()
hilo_ia.start()

ventana.after(0, actualizar_interfaz)

print("Sistema de control vehicular iniciado.")

ventana.mainloop()