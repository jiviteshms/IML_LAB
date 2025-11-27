import tensorflow as tf
from keras import layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

def build_cnn(filter_size=3, activation='relu', pooling='max'):
    model = models.Sequential()

    model.add(layers.Conv2D(16, (filter_size, filter_size), activation=activation, input_shape=(28, 28, 1)))
    if pooling == 'max':
        model.add(layers.MaxPooling2D((2, 2)))
    else:
        model.add(layers.AveragePooling2D((2, 2)))

    model.add(layers.Conv2D(32, (filter_size, filter_size), activation=activation))
    if pooling == 'max':
        model.add(layers.MaxPooling2D((2, 2)))
    else:
        model.add(layers.AveragePooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

configs = [
    {'filter_size': 2, 'activation': 'relu', 'pooling': 'max'},
    {'filter_size': 3, 'activation': 'relu', 'pooling': 'max'},
    {'filter_size': 4, 'activation': 'sigmoid', 'pooling': 'avg'},
    {'filter_size': 5, 'activation': 'sigmoid', 'pooling': 'avg'}
]

results = []

for config in configs:
    print(f"\nTraining model with config: {config}")
    model = build_cnn(**config)

    history = model.fit(train_images, train_labels, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

    train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

    y_pred = np.argmax(model.predict(test_images, verbose=0), axis=1)
    cm = confusion_matrix(test_labels, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {config}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    results.append({
        'config': config,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    })

print("\nSummary of Results:")
for r in results:
  print(r)

