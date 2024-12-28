TRAINING_FOLDER: str = "C:/Users/usuario/PycharmProjects/cnn_img/input/"
TRAIN_DIR: str = f"{TRAINING_FOLDER}/train"
VAL_DIR: str = f"{TRAINING_FOLDER}/val"
TEST_DIR: str = f"{TRAINING_FOLDER}/test"
MODEL_OUTPUT: str = "C:/Users/usuario/PycharmProjects/cnn_img/models/"

# Configuraciones de datos
TAMANO_IMAGEN = 256  # Tamaño de las imágenes para entrenamiento y validación
CLASES = ["NORMAL", "PNEUMONIA"]

# Configuraciones de entrenamiento
EPOCHS = 15  # Número de épocas
BATCH_SIZE = 32  # Tamaño del batch
EFFECTIVE_BATCH_SIZE = 128  # Tamaño efectivo del batch
ACCUMULATION_STEPS = max(1, EFFECTIVE_BATCH_SIZE // BATCH_SIZE)  # Pasos de acumulación

# Configuraciones del optimizador
LEARNING_RATE = 0.0007
WEIGHT_DECAY = 1e-4

# Configuraciones del scheduler
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2

# Configuraciones de Focal Loss
FOCAL_LOSS_ALPHA = [1.0]  # Pesos para las clases
FOCAL_LOSS_GAMMA = 3  # Parámetro gamma

# Configuración de Dropout adaptativo
DROPOUT_BASE_RATE = 0.3  # Tasa inicial de dropout
DROPOUT_DECAY = 0.0  # Decaimiento de la tasa de dropout por época
DROPOUT_MIN_RATE = 0.01  # Tasa mínima de dropout



