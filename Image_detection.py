import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

# Step 1: Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the images to scale pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the class names for CIFAR-10 (10 categories)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Step 2: Define the Convolutional Neural Network (CNN) architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # First convolutional layer
    layers.MaxPooling2D((2, 2)),  # First pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    layers.MaxPooling2D((2, 2)),  # Second pooling layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Third convolutional layer
    layers.Flatten(),  # Flatten the feature maps
    layers.Dense(64, activation='relu'),  # Fully connected layer with 64 neurons
    layers.Dense(10)  # Output layer (10 classes for CIFAR-10)
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Step 4: Train the model
print("Training the model...")
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Step 5: Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

# Step 6: Function to upload an image and predict its class
def upload_and_predict(model, class_names):
    # Hide the Tkinter GUI window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Open file dialog to upload an image
    print("Please select an image to classify...")
    image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    
    if not image_path:
        print("No file selected. Exiting...")
        return
    
    # Load the image and resize it to match the input size of the model (32x32)
    img = load_img(image_path, target_size=(32, 32))
    img_array = img_to_array(img)  # Convert the image to a NumPy array
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    
    # Predict the class of the uploaded image
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    
    # Display the uploaded image along with the predicted class
    plt.imshow(load_img(image_path))
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}")
    plt.show()
    
    print(f"The uploaded image is predicted to be: {predicted_class}")

# Step 7: Call the function to classify an uploaded image
upload_and_predict(model, class_names)
