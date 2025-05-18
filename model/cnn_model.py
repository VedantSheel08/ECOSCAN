import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class EcoscanCNN:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.classes = ['Animal', 'Fire/Smoke', 'Trash', 'Water Body', 'Plant', 'Terrain']
        self.num_classes = len(self.classes)
        self.model = self._build_model()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)

    def _build_model(self):
        """
        Build the CNN architecture
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def preprocess_image(self, image_path):
        """
        Preprocess a single image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        img = img / 255.0  # Normalize pixel values
        return img

    def load_data(self, data_dir):
        """
        Load and preprocess the dataset
        """
        images = []
        labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Directory {class_path} not found")
                continue
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = self.preprocess_image(img_path)
                    images.append(img)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
        
        X = np.array(images)
        y = self.label_encoder.transform(labels)
        y = tf.keras.utils.to_categorical(y, self.num_classes)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model
        """
        # Data augmentation for training
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        # Create early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Model checkpoint callback
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True
        )

        # Train the model
        history = self.model.fit(
            data_augmentation(X_train),
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint_cb]
        )
        
        return history

    def predict(self, image_path):
        """
        Make prediction on a single image
        """
        img = self.preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img)
        predicted_class = self.classes[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        
        return predicted_class, confidence

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_loss, test_accuracy

    def save_model(self, filepath):
        """
        Save the model
        """
        self.model.save(filepath)

    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model
        """
        instance = cls()
        instance.model = tf.keras.models.load_model(filepath)
        return instance 