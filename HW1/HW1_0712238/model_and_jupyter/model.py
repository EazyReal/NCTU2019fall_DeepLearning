import numpy as np

class NN(object):
    def __init__(self, layers = [10 , 20, 1], activations=['sigmoid', 'relu'], usage = 'regression'):
        #my model is very sensitive to initial value sometimes
        #maybe the gradient descent method is too naive
        assert(len(layers) == len(activations)+1)
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        self.usage = usage
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1], layers[i])*0.1) #the *0.1 is impmortant
            self.biases.append(np.random.randn(layers[i+1], 1)*0.1)

    def feedforward(self, x): #x = dim*num
        ai = np.copy(x)
        z_s = []
        a_s = [ai]
        for i in range(len(self.weights)):
            #activation_function = self.AF(self.activations[i])
            z_s.append(self.weights[i].dot(ai) + self.biases[i])
            ai = self.AF(self.activations[i])(z_s[-1])
            a_s.append(ai)
        return (z_s, a_s)

    def backpropagation(self,y, z_s, a_s): #y = 1*num
        dw = []  # dC/dW
        db = []  # dC/dB
        deltas = [None] * len(self.weights)  # delta = dC/dZ, error for each layer

        #out delta measurement =
        delta_out = y- a_s[-1]
        #last layer delta
        deltas[-1] = delta_out*(self.dAF(self.activations[-1]))(z_s[-1])
        #backpro
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.dAF(self.activations[i])(z_s[i]))
        batch_size = y.shape[1]
        db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
        #db = [d.dot(np.ones((batch_size,1))) for d in deltas]
        #dw = [d.dot(a_s[i].T) for i,d in enumerate(deltas)]
        # return the derivitives respect to weight matrix and biases
        #print(db)
        #print(dw)
        return dw, db

    def train(self, x, y, batch_size=10, epochs=100, lr = 0.1): #x = num*dim #y = num*dim
        #record cost by epchos
        learning_curve = []

        #mini batch
        #assert(x.shape[0] >= batch_size*epochs)
        indices = np.arange(x.shape[0])#debug if 0
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        for e in range(epochs):
            i=0
            #print("len y  ", len(y))
            while(i<len(y)):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                x_batch = x_batch.T
                y_batch = y_batch.T
                #print(x_batch.shape)
                #print(y_batch.shape)
                i += batch_size
                z_s, a_s = self.feedforward(x_batch)
                dw, db = self.backpropagation(y_batch, z_s, a_s)
                self.weights = [wi+lr*dwi for wi,dwi in  zip(self.weights, dw)]
                self.biases = [bi+lr*dbi for bi,dbi in  zip(self.biases, db)]
                loss = self.J(self.usage)(a_s[-1],y_batch)
            #if(e%(epochs/10)== 0):
            learning_curve.append(loss) #to expand
            #print("loss = {}".format(np.linalg.norm(a_s[-1]-y_batch))) #to expand
        return learning_curve


    def calc_error(self, test_X, test_y): #num*dim
        _, a_s = self.feedforward(test_X.T)
        return  self.J(self.usage)(a_s[-1], test_y.T)

    def prediction(self, X): #num*dim
        _, a_s = self.feedforward(X.T)
        return a_s[-1]

    @staticmethod
    def AF(name):
        if(name == 'sigmoid'):
            def sig(x):
                x = np.clip(x , -500, 500)
                return np.exp(x)/(1+np.exp(x))
            return sig
        elif(name == 'linear'):
            return lambda x : x
        elif(name == 'relu'):
            def relu(x):
                y = np.copy(x)
                y[y<0] = 0
                return y
            return relu
        else:
            print('unknown activation function => linear')
            return lambda x: x

    @staticmethod
    def dAF(name):
        if(name == 'sigmoid'):
            def dsig(x):
                x = np.clip(x , -500, 500)
                sigx = np.exp(x)/(1+np.exp(x))
                return sigx*(1-sigx)
            return dsig
        elif(name == 'linear'):
            return lambda x: 1
        elif(name == 'relu'):
            def drelu(x):
                y = np.copy(x)
                y[y>=0] = 1
                y[y<0] = 0
                return y
            return drelu
        else:
            print('unknown activation function => linear derivative')
            return lambda x: 1

    @staticmethod
    def dJ(name):
        if(name == 'regression'):
            return lambda x, y: y-x
        if(name == 'classification'):
            return lambda x, y: np.divide(y, x) - np.divide(1 - y, 1 - x)
        else:
            print('unknown usage => regression')
            return lambda x, y: y-x

    @staticmethod
    def J(name):
        if(name == 'regression'):
            return lambda x, y: np.sqrt(np.linalg.norm(y-x)/max(y.shape[0], y.shape[1])) #RMS
        if(name == 'classification'):
            return lambda x, y: np.divide(y, x) - np.divide(1 - y, 1 - x)
        else:
            print('unknown usage => regression')
            return lambda x, y: np.sqrt(np.linalg.norm(y-x)/max(y.shape[0], y.shape[1])) #RMS
