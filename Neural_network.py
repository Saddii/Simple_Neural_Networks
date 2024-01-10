import numpy as np
import pathlib

class NeuralNetwork:
    def __init__(self,learning_rate =0.01):
        self.learning_rate = learning_rate

        with np.load(f"{pathlib.Path(__file__).parent.absolute()}/mnist.npz") as f:
            self.train_X,self.train_y= f["x_train"]/255, f["y_train"]
            self.test_X,self.test_y=f["x_test"],f["y_test"]

    def fit(self,neurons_in_layer=(784,20,10),Epochs=3,c=0.01):
        
        self.train_y = np.eye(10)[self.train_y]

        layers = len(neurons_in_layer)
        obs_num, n , m = self.train_X.shape  #60k Observations  28x28 matrix
        X = self.train_X.reshape(obs_num, n*m,1) #60k vectors len = 784
        W = [] 

        for w in range(layers-1):
            W.append(np.random.uniform(-0.5,0.5,(neurons_in_layer[w+1],neurons_in_layer[w]+1))) # +1 ->bias
        
        correct = 0
        for j in range(Epochs):
            for i in range(obs_num):
                
                #forwardporopagation
                X_0 = np.insert(X[i],0,1) #+ bias 
                net_0 = W[0]@X_0 #vals of neurons before activation in first layer
                
                X_1 = 1/(1+np.exp(-net_0)) #val of neurons after activation in first layer
                X_1 = np.insert(X_1,0,1) #+ bias
                net_1 = W[1]@X_1 #vals of neurons before activation in output layer
                a = 1/(1+np.exp(-net_1)) #activation in output layer
                correct += int(np.argmax(a)==np.argmax(self.train_y[i]))

                #backpropagation
                #loss = (a-self.train_y)^2
                L = a - self.train_y[i] # Loss deriv
                d_1= L
                L_W_1 = L.reshape(10,1) @X_1.reshape(1,-1)
                W[1] += -c*L_W_1
                L_X_1 = (W[1]).T @ d_1

                L_a_0 =L_X_1[1:]
                d_0 = L_a_0 *X_1[1:]*(1-X_1[1:])
                L_W_0=d_0.reshape(d_0.shape[0],1) @X_0.reshape(1,-1)
                W[0] += -c*L_W_0
                
                
            print("Training Accuracy:",correct/60000)
            correct = 0
        return (W[0],W[1])
    def evaluate(self,W0,W1):
            test_obs_num = self.test_X.shape[0]
            test_X = self.test_X.reshape(test_obs_num, -1, 1)
            test_y = np.eye(10)[self.test_y]

            correct = 0
            for i in range(test_obs_num):
                X_0 = np.insert(test_X[i], 0, 1)
                net_0 = W0 @ X_0

                X_1 = 1 / (1 + np.exp(-net_0))
                X_1 = np.insert(X_1, 0, 1)
                net_1 = W1 @ X_1
                a = 1 / (1 + np.exp(-net_1))
                correct += int(np.argmax(a) == np.argmax(test_y[i]))

            accuracy = correct / test_obs_num
            print("Test Accuracy:", accuracy)

test = NeuralNetwork()
W0,W1=test.fit()
test.evaluate(W0,W1)
    
