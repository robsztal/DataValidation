from tensorflow.keras.datasets import fashion_mnist
from PIL import Image

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#check images
if False:
    for i in range(5):
        img = Image.fromarray(X_test[i])
        img.save("//Users/rob/Documents/GitHub/DataValidation/uploads/{}.png".format(i))
