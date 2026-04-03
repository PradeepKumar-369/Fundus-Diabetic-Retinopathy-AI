import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# ==========================================
# 1. LOAD DATASET (Loaded once to save time)
# ==========================================
train_path = "data/augmented_resized_V2/train"
val_path   = "data/augmented_resized_V2/val"

print("Loading datasets...")
train_dataset = tf.keras.utils.image_dataset_from_directory(train_path, image_size=(224, 224), batch_size=32)
val_dataset = tf.keras.utils.image_dataset_from_directory(val_path, image_size=(224, 224), batch_size=32)

# Create very small subsets for GWO so it doesn't take 10 hours to run
gwo_train_data = train_dataset.take(50) 
gwo_val_data   = val_dataset.take(10)

# ==========================================
# 2. THE FITNESS FUNCTION (GWO + EfficientNet)
# ==========================================
def fitness_function(hyperparameters):
    # GWO gives us these numbers to test
    learning_rate = hyperparameters[0]
    dropout_rate = hyperparameters[1]
    
    print(f"\n--> Testing Model -> LR: {learning_rate:.5f} | Dropout: {dropout_rate:.2f}")
    
    # 1. Build Model using GWO's Dropout Rate
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)  # <--- GWO CONTROLS THIS NOW
    outputs = layers.Dense(5, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # 2. Compile Model using GWO's Learning Rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # <--- GWO CONTROLS THIS NOW
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # 3. Train the model for 1 epoch to test these settings
    history = model.fit(gwo_train_data, validation_data=gwo_val_data, epochs=1, verbose=1)
    
    # 4. Get the validation accuracy
    val_acc = history.history['val_accuracy'][-1]
    
    # GWO minimizes the score, so we return Error (1 - Accuracy)
    return 1.0 - val_acc

# ==========================================
# 3. THE GREY WOLF OPTIMIZER ALGORITHM
# ==========================================
def gwo_optimize(num_wolves, max_iter, lower_bounds, upper_bounds):
    dimensions = len(lower_bounds)
    
    alpha_pos = np.zeros(dimensions)
    alpha_score = float("inf")
    
    beta_pos = np.zeros(dimensions)
    beta_score = float("inf")
    
    delta_pos = np.zeros(dimensions)
    delta_score = float("inf")
    
    positions = np.zeros((num_wolves, dimensions))
    for i in range(num_wolves):
        for j in range(dimensions):
            positions[i, j] = random.uniform(lower_bounds[j], upper_bounds[j])
            
    for t in range(max_iter):
        print(f"\n======================================")
        print(f" GWO ITERATION {t+1} OF {max_iter} ")
        print(f"======================================")
        
        for i in range(num_wolves):
            # Keep wolves inside boundaries
            for j in range(dimensions):
                positions[i, j] = np.clip(positions[i, j], lower_bounds[j], upper_bounds[j])
                
            # Train the model with these settings
            fitness = fitness_function(positions[i])
            
            # Update Leaders
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i].copy()
                
        # Move pack toward leaders
        a = 2 - t * (2 / max_iter)
        for i in range(num_wolves):
            for j in range(dimensions):
                r1, r2 = random.random(), random.random()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                r1, r2 = random.random(), random.random()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                r1, r2 = random.random(), random.random()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                positions[i, j] = (X1 + X2 + X3) / 3

    return alpha_pos, alpha_score

# ==========================================
# 4. EXECUTE & SAVE FINAL MODEL
# ==========================================
if __name__ == "__main__":
    # GWO Settings: Start small! 3 wolves x 2 iterations = 6 model trainings
    WOLVES = 3       
    ITERATIONS = 2   
    
    # Boundaries: [Learning Rate, Dropout Rate]
    LB = [0.0001, 0.1] 
    UB = [0.01, 0.5]   
    
    print("Starting Grey Wolf Optimization...")
    best_hyperparams, best_error = gwo_optimize(WOLVES, ITERATIONS, LB, UB)
    
    print("\n######################################")
    print(" GWO OPTIMIZATION COMPLETE")
    print(f" Best Learning Rate: {best_hyperparams[0]:.5f}")
    print(f" Best Dropout Rate:  {best_hyperparams[1]:.2f}")
    print(f" Best Val Accuracy:  {(1.0 - best_error) * 100:.2f}%")
    print("######################################\n")
    
    # --- NOW TRAIN THE FINAL MODEL WITH THE BEST SETTINGS ---
    print("Training FINAL model with optimal GWO settings...")
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(best_hyperparams[1])(x) 
    outputs = layers.Dense(5, activation="softmax")(x)
    
    final_model = models.Model(inputs=inputs, outputs=outputs)
    final_optimizer = tf.keras.optimizers.Adam(learning_rate=best_hyperparams[0])
    
    final_model.compile(optimizer=final_optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Train fully (using more data)
    final_train = train_dataset
    final_val   = val_dataset
    
    final_model.fit(final_train, validation_data=final_val, epochs=3)
    
    final_model.save("models/dr_model.keras")
    print("\nSUCCESS: Final GWO-Optimized model saved as models/dr_model.keras!")