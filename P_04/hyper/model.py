
import numpy as np
import tensorflow as tf
import os

def hyperparameter_tun(d1,d2,lr,epochs,batch):
    """
    Esta función crea y entrena un modelo utilizando los hiperparámetros proporcionados.

    :param d1: Número de neuronas en la primera capa densa.
    :type d1: int
    :param d2: Número de neuronas en la segunda capa densa.
    :type d2: int
    :param lr: Tasa de aprendizaje para el optimizador Adam.
    :type lr: float
    :param epochs: Número de épocas para entrenar el modelo.
    :type epochs: int
    :param batch: Tamaño del lote para el entrenamiento.
    :type batch: int
    :return: La precisión de validación del modelo entrenado en la última época.
    :rtype: float
    """
    # Subclassing
    class MyModel(tf.keras.Model):  # Hereda de tf.keras.Model.
        def __init__(self, d2):
            super(MyModel, self).__init__()
            self.dense = tf.keras.layers.Dense(d2)

        def call(self, x):
            return self.dense(x)

    class MyModel2(MyModel):  # Hereda de MyModel.
        def __init__(self, d1=10,d2=10):
            super(MyModel2, self).__init__(d2)
            self.dense2 = tf.keras.layers.Dense(d1)
            self.dense3 = tf.keras.layers.Dense(1)

        def call(self, x):
            x = super().call(x)  # Llama al método call de MyModel.
            x = self.dense2(x)
            return self.dense3(x)
        
    model = MyModel2(d1,d2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='MSE', metrics=['accuracy'])
    
    x_train = np.random.random((1000, 10))
    y_train = np.random.randint(2, size=(1000,))
    x_val = np.random.random((200, 10))
    y_val = np.random.randint(2, size=(200,))

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch, validation_data=(x_val, y_val))

    hp_metric = history.history['val_accuracy'][-1]
    # Evaluar el modelo
    x_test = np.random.random((200, 10))
    y_test = np.random.randint(2, size=(200,))
    evv = model.evaluate(x_test, y_test)
    print("HERE: ",evv)
    model.save('model/m1')
    return hp_metric
