import tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow.keras import layers\n\n\nclass CNNModel:  
    def __init__(self):  
        self.model = self.build_model()  
  
    def build_model(self):  
        model = keras.Sequential()  
        model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(None, 1)))  
        model.add(layers.MaxPooling1D(pool_size=2))  
        model.add(layers.Flatten())  
        model.add(layers.Dense(64, activation='relu'))  
        model.add(layers.Dense(1, activation='sigmoid'))  
        return model  
  
    def compile_model(self):  
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
  
    def train_model(self, X_train, y_train, epochs=10):  
        self.model.fit(X_train, y_train, epochs=epochs)  
\n\nclass DenseModel:  
    def __init__(self):  
        self.model = self.build_model()  
  
    def build_model(self):  
        model = keras.Sequential()  
        model.add(layers.Input(shape=(None,)))  
        model.add(layers.Dense(64, activation='relu'))  
        model.add(layers.Dense(32, activation='relu'))  
        model.add(layers.Dense(1, activation='sigmoid'))  
        return model  
  
    def compile_model(self):  
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
  
    def train_model(self, X_train, y_train, epochs=10):  
        self.model.fit(X_train, y_train, epochs=epochs)  
\n\nclass HybridModel:  
    def __init__(self):  
        self.model = self.build_model()  
  
    def build_model(self):  
        cnn_input = layers.Input(shape=(None, 1))  
        cnn_out = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(cnn_input)  
        cnn_out = layers.MaxPooling1D(pool_size=2)(cnn_out)  
        cnn_out = layers.Flatten()(cnn_out)  
        dense_input = layers.Input(shape=(None,))  
        dense_out = layers.Dense(64, activation='relu')(dense_input)  
        merged = layers.concatenate([cnn_out, dense_out])  
        final_output = layers.Dense(1, activation='sigmoid')(merged)  
        model = keras.Model(inputs=[cnn_input, dense_input], outputs=final_output)  
        return model  
  
    def compile_model(self):  
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
  
    def train_model(self, X_train_cnn, X_train_dense, y_train, epochs=10):  
        self.model.fit([X_train_cnn, X_train_dense], y_train, epochs=epochs)  
