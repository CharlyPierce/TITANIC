import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Modelo Secuencial
model_sequential = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])

strategy = tf.distribute.get_strategy() 
#strategy = tf.distribute.MirroredStrategy()  # varias GPUs en una máquina.
# Abrir un scope de estrategia
with strategy.scope():
    model = model_sequential

    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Generar algunos datos de ejemplo
x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000,))

# Crear un callback de ModelCheckpoint
checkpoint = ModelCheckpoint('gs://models_ai_save/best_model',
                             monitor='accuracy',  # la métrica a monitorizar
                             verbose=1,  # log level
                             save_best_only=True,  # solo guarda el mejor modelo
                             mode='max',  # modo 'max' porque queremos maximizar la accuracy
                             save_format='tf')  # especifica que se debe usar el formato SavedModel

# Entrenar el modelo
model.fit(x_train, y_train, epochs=3, batch_size=32, callbacks=[checkpoint])

# Evaluar el modelo
x_test = np.random.random((200, 10))
y_test = np.random.randint(2, size=(200,))
evv = model.evaluate(x_test, y_test)
print(evv)

# Guardar el modelo final en el bucket de GCS
model.save('gs://models_ai_save/model_final')
