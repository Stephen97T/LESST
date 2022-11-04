from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from models import RNN


dataset = loadtxt(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    delimiter=",",
)
X = dataset[:, 0:8]
y = dataset[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y)
inputdim = X_train.shape[1]
outputdim = 1
rnn = RNN(inputdim, outputdim)
model = Sequential()
model.add(Dropout(0.15, input_shape=(8,)))
model.add(Dense(12, activation="relu"))
# model.add(Dropout(0.1))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)
history = model.fit(
    X_train,
    y_train,
    epochs=1500,
    batch_size=20,
    validation_data=(X_test, y_test),
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

import matplotlib.pyplot as plt

epochs = range(11, len(acc) + 1)

plt.plot(epochs, acc[10:], "r", label="Training acc")
plt.plot(epochs, val_acc[10:], "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.legend()

plt.figure()

plt.plot(epochs, loss[10:], "r", label="Training loss")
plt.plot(epochs, val_loss[10:], "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()
