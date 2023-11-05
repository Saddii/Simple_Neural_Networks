from a import get_mnist
import numpy as np
from matplotlib import pyplot
import pickle
import pathlib

def function(x): #funkcja aktywacji
    return 1/(1+np.exp(-x))
def fun_deriv(x):
    return function(x)*(1-function(x))
fun = np.vectorize(function)
fun_d = np.vectorize(fun_deriv)
c=0.01
with np.load(f"{pathlib.Path(__file__).parent.absolute()}/mnist.npz") as f:
    train_X,train_y= f["x_train"], f["y_train"]
    test_X,test_y=f["x_test"],f["y_test"]

train_X =train_X.astype("float32") / 255
np.reshape(train_X, (train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
train_y = np.eye(10)[train_y]


with open('W_1.pkl', 'rb') as f:
    W_1 = pickle.load(f)
with open('W_2.pkl', 'rb') as f:
    W_2 = pickle.load(f)

# print(train_X.shape,train_y.shape)
# W_1 = np.random.uniform(-0.5,0.5,(100,785))
# W_2 = np.random.uniform(-0.5,0.5,(10,101))

X = train_X.reshape(60000,784,1)

def learn(W_1,W_2,X):
    prawidlowe = 0 
    i=0
    for j in range(3):
        for i in range(60000):
            X_1 = np.insert(X[i],0,1)
            net_1 = W_1@X_1
            X_2 = 1/(1+np.exp(-net_1))
            X_2 = np.insert(X_2,0,1)
            net_2 = W_2@X_2
            a = 1/(1+np.exp(-net_2))
            prawidlowe+= int(np.argmax(a)==np.argmax(train_y[i]))
            # if np.argmax(a)!=np.argmax(train_y[i]):
                
            #     img = train_X[i]
            #     pyplot.imshow(img.reshape(28,28),cmap="Greys")
            #     pyplot.show()


            
            L= a-train_y[i]
            d_2= L
            L_W_2 = L.reshape(10,1) @(X_2.reshape(1,X_2.shape[0]))
            W_2 += -c*L_W_2
            L_X_2 = (W_2).T @ d_2

            L_a_1 =L_X_2[1:]
            d_1 = L_a_1 *X_2[1:]*(1-X_2[1:])
            L_W_1=d_1.reshape(d_1.shape[0],1) @X_1.reshape(1,X_1.shape[0])
            W_1 += -c*L_W_1
        print("Prawidlowe:",prawidlowe/60000)
        prawidlowe=0
    return (W_1,W_2)

ret = learn(W_1,W_2,X)
W_1=ret[0]
W_2=ret[1]

with open('W_1.pkl', 'wb') as f:
    pickle.dump(W_1, f)

# Zapisz W_2 do pliku pickle
with open('W_2.pkl', 'wb') as f:
    pickle.dump(W_2, f)
    
while True:
    ask = input("kontynuowac uczenie? T/N ")
    if ask == "T":
        ret = learn(W_1,W_2,X)
        W_1=ret[0]
        W_2=ret[1]

        with open('W_1.pkl', 'wb') as f:
            pickle.dump(W_1, f)

        # Zapisz W_2 do pliku pickle
        with open('W_2.pkl', 'wb') as f:
            pickle.dump(W_2, f)
    
    try:
        index = int(input("Liczba 0-59999 "))
    except:
        next
    img = train_X[index]
    pyplot.imshow(img.reshape(28,28),cmap="Greys")

    X_1 = np.insert(X[index],0,1)
    net_1 = W_1@X_1
    X_2 = fun(net_1)
    X_2 = np.insert(X_2,0,1)
    net_2 = W_2@X_2
    a = fun(net_2)
    print(np.argmax(a),train_y[index])
    pyplot.show()
        
        
