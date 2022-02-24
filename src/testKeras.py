from keras.datasets import mnist
# Bibliothek für grafische Darstellung laden
import matplotlib.pyplot as plt
# Funktion für zufällige Bildauswahl laden
from random import randint

# Datensätze laden
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Bild zeigen
plt.figure()
plt.imshow(train_images[randint(1, len(train_images) - 1)])
plt.grid(False)
plt.show()