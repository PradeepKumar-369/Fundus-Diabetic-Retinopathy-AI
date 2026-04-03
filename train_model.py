import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# -------------------
# Load Dataset
# -------------------
train_path = "data/augmented_resized_V2/train"
val_path   = "data/augmented_resized_V2/val"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=(224, 224),
    batch_size=32
)

# -------------------
# Functional Model (IMPORTANT)
# -------------------
inputs = tf.keras.Input(shape=(224, 224, 3))

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=inputs
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)

# -------------------
# Compile
# -------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------
# Train Small Subset
# -------------------
small_train = train_dataset.take(200)
small_val   = val_dataset.take(50)

model.fit(
    small_train,
    validation_data=small_val,
    epochs=1
)

# -------------------
# Save Model (Modern Format)
# -------------------
model.save("models/dr_model.keras")

print("Model saved successfully!")
