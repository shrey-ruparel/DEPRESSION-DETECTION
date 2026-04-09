import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# =========================
# PATHS
# =========================
DATASET_PATH = r"C:\Users\rutuja\Desktop\dep_detection\DEPRESSION-DETECTION\BACKEND\Data\facedetection\train"
MODEL_SAVE_PATH = r"C:\Users\rutuja\Desktop\dep_detection\DEPRESSION-DETECTION\BACKEND\MODELS\emotion_model_best.h5"

IMG_SIZE = 160   # 🔥 smaller = faster
BATCH_SIZE = 16  # 🔥 lower RAM
EPOCHS_INITIAL = 10
EPOCHS_FINE = 5

# =========================
# LOAD DATASET
# =========================
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

print("Classes:", train_data.class_names)

# =========================
# OPTIMIZATION (NO CACHE ❌)
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.shuffle(500).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# =========================
# DATA AUGMENTATION
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =========================
# MODEL (MobileNetV2)
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # freeze

# =========================
# BUILD MODEL
# =========================
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(7, activation='softmax')(x)

model = models.Model(inputs, outputs)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# CALLBACKS
# =========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True
    )
]

# =========================
# TRAIN PHASE 1
# =========================
print("\n🔥 Training (Lightweight)...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_INITIAL,
    callbacks=callbacks
)

# =========================
# FINE-TUNING (LIGHT)
# =========================
print("\n🔥 Fine-tuning...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

# =========================
# SAVE MODEL
# =========================
model.save(MODEL_SAVE_PATH)

print("\n✅ Training Complete!")
print("Saved at:", MODEL_SAVE_PATH)