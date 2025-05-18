import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps  # Install pillow instead of PIL
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Custom DepthwiseConv2D layer that ignores the groups parameter
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        if 'groups' in config:
            del config['groups']
        return cls(**config)

class RealtimeClassifier:
    def __init__(self):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)
        
        try:
            # Define custom objects for model loading
            custom_objects = {
                'DepthwiseConv2D': CustomDepthwiseConv2D
            }
            
            # Load the model with custom objects
            with tf.keras.utils.custom_object_scope(custom_objects):
                self.model = tf.keras.models.load_model("converted_keras (16)/keras_model.h5", compile=False)
            
            # Load the labels
            with open("converted_keras (16)/labels.txt", "r") as f:
                self.class_names = f.readlines()
            print("Model loaded successfully!")
            print(f"Available categories: {[name.strip() for name in self.class_names]}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

        # Standard image size for Teachable Machine models
        self.image_size = (224, 224)
        
        # Define colors for each class (in BGR format)
        self.colors = {
            'Animal': (0, 255, 0),    # Green
            'fire': (0, 0, 255),      # Red
            'Trash': (255, 0, 0),     # Blue
            'Plants': (0, 255, 255)    # Yellow
        }

    def preprocess_frame(self, frame):
        """Preprocess frame for model input using the exact Teachable Machine approach."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(rgb_frame)
        
        # Resize and crop from center
        image = ImageOps.fit(image, self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.asarray(image)
        
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        
        # Create the array of the right shape
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        return data

    def predict(self, frame):
        """Make prediction on a frame."""
        # Preprocess the frame
        data = self.preprocess_frame(frame)
        
        # Get prediction
        prediction = self.model.predict(data, verbose=0)
        index = np.argmax(prediction[0])
        class_name = self.class_names[index].strip()
        confidence_score = float(prediction[0][index])
        
        # Remove the leading number and space if present (e.g., "1. Label" -> "Label")
        if '. ' in class_name:
            class_name = class_name.split('. ')[1]
        
        return class_name, confidence_score

class EcoScanGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("ECOSCAN Classifier")
        self.window.geometry("1200x800")
        
        # Initialize classifier
        self.classifier = RealtimeClassifier()
        
        # Create GUI elements
        self.create_widgets()
        
        # Video capture variables
        self.cap = None
        self.is_running = False
        self.last_frame_time = time.time()
        self.fps = 0
        
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame (left side)
        self.video_frame = ttk.Frame(main_frame)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel (right side)
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Title
        title_label = ttk.Label(control_frame, text="ECOSCAN Controls", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Start/Stop button
        self.toggle_button = ttk.Button(control_frame, text="Start", command=self.toggle_camera)
        self.toggle_button.pack(fill=tk.X, pady=5)
        
        # Confidence threshold
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(threshold_frame, text="Confidence Threshold:").pack()
        self.threshold_var = tk.DoubleVar(value=0.6)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, 
                                  variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X)
        
        # Results display
        results_frame = ttk.LabelFrame(control_frame, text="Detection Results", padding=10)
        results_frame.pack(fill=tk.X, pady=10)
        
        self.class_label = ttk.Label(results_frame, text="Class: -", font=('Arial', 12))
        self.class_label.pack(fill=tk.X)
        
        self.confidence_label = ttk.Label(results_frame, text="Confidence: -", font=('Arial', 12))
        self.confidence_label.pack(fill=tk.X)
        
        self.fps_label = ttk.Label(results_frame, text="FPS: -", font=('Arial', 12))
        self.fps_label.pack(fill=tk.X)
        
    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return
        
        self.is_running = True
        self.toggle_button.configure(text="Stop")
        self.update_frame()
    
    def stop_camera(self):
        self.is_running = False
        self.toggle_button.configure(text="Start")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def update_frame(self):
        if self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Calculate FPS
                current_time = time.time()
                self.fps = 1 / (current_time - self.last_frame_time)
                self.last_frame_time = current_time
                
                # Make prediction
                label, confidence = self.classifier.predict(frame)
                
                # Update GUI labels
                self.class_label.configure(text=f"Class: {label if label else '-'}")
                self.confidence_label.configure(text=f"Confidence: {confidence*100:.1f}%" if confidence else "Confidence: -")
                self.fps_label.configure(text=f"FPS: {self.fps:.1f}")
                
                # Add visualization
                if label and confidence > self.threshold_var.get():
                    color = self.classifier.colors.get(label, (0, 255, 0))
                    height, width = frame.shape[:2]
                    
                    # Draw bounding box
                    box_width = int(width * 0.9)
                    box_height = int(height * 0.9)
                    x = (width - box_width) // 2
                    y = (height - box_height) // 2
                    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 2)
                    
                    # Add label
                    text = f"{label}: {confidence:.1%}"
                    cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = ImageTk.PhotoImage(image=img)
                
                self.video_label.configure(image=img)
                self.video_label.image = img
            
            self.window.after(10, self.update_frame)
    
    def on_closing(self):
        self.stop_camera()
        self.window.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = EcoScanGUI(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}") 