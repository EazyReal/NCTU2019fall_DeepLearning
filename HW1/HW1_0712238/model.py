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
            if self.activations[i] in ['relu', 'selu', 'elu']:
                self.weights.append(np.random.randn(layers[i+1], layers[i])*np.sqrt(2./layers[i])) #heuristic
                self.biases.append(np.random.randn(layers[i+1], 1)*0.1) #can be 0
            else:
                self.weights.append(np.random.randn(layers[i+1], layers[i])*np.sqrt(1./layers[i])) #heuristic
                self.biases.append(np.random.randn(layers[i+1], 1)*0.1) #can be 0


    def feedforward(self, x): #x = dim*num
        ai = np.copy(x)
        z_s = []
        a_s = [ai]
        for i in range(len(self.weights)):
            z_s.append(self.weights[i].dot(ai) + self.biases[i])
            ai = self.AF(self.activations[i])(z_s[-1])
            a_s.append(ai)
        return z_s, a_s

    def backpropagation(self,y, z_s, a_s): #y = 1*num
        dw = []  # dJ/dW
        db = []  # dJ/dB
        deltas = [None] * len(self.weights)  # delta = dJ/dZ, error for each layer

        #delta out
        delta_out = self.dJ(self.usage)(a_s[-1], y)
        #last layer delta
        deltas[-1] = delta_out*(self.dAF(self.activations[-1]))(z_s[-1])
        #backpro
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.dAF(self.activations[i])(z_s[i]))
        batch_size = y.shape[1]
        db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]

        eps = 0.001
        #for i in range(len(dw)):
        #    assert(np.linalg.norm(dw[i]) > eps)
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

    def calc_accuracy(self, test_X, test_y): #num*dim
        _, a_s = self.feedforward(test_X.T)
        n = a_s[-1].shape[1]
        total = 0.
        correct = 0.
        for i in range(n):
            total += 1
            if (a_s[-1][0][i] >= 0.5) == bool(test_y[i]):
                correct += 1
        return  correct/total

    @staticmethod
    def AF(name):
        if(name == 'sigmoid'):
            def sig(x):
                x = np.clip(x , -10., 10.)
                return np.exp(x)/(1+np.exp(x))
            return sig
        elif(name == 'linear'):
            return lambda x : x
        elif(name == 'relu'):
            def relu(x):
                return np.where(x<0,0,x)
            return relu
        elif(name == 'selu'):
            def selu(x,lamb=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
                x = np.clip(x , -10., 10.)
                return lamb*np.where(x<0,alpha*(np.exp(x) - 1),x)
            return selu
        else:
            print('unknown activation function => linear')
            return lambda x: x

    @staticmethod
    def dAF(name):
        if(name == 'sigmoid'):
            def dsig(x):
                x = np.clip(x , -10., 10.)
                sigx = np.exp(x)/(1+np.exp(x))
                return sigx*(1-sigx)
            return dsig
        elif(name == 'linear'):
            return lambda x: 1
        elif(name == 'relu'):
            def drelu(x):
                return np.where(x<0,0,1)
            return drelu
        elif(name == 'selu'):
            def dselu(x,lamb=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
                x = np.clip(x , -10., 10.)
                return lamb*np.where(x<0,alpha*np.exp(x),1)
            return dselu
        else:
            print('unknown activation function => linear derivative')
            return lambda x: 1

    @staticmethod
    def dJ(name):
        if(name == 'regression'):
            return lambda x, y: y-x
        if(name == 'classification'):
            def dCE(yhat, y):
                epsilon=1e-12
                yhat = np.clip(yhat, epsilon, 1. - epsilon)
                return np.divide(y, yhat) - np.divide(1 - y, 1 - yhat)
            return dCE
        else:
            print('unknown usage => regression')
            return lambda x, y: y-x

    @staticmethod
    def J(name):
        if(name == 'regression'):
            return lambda x, y: np.linalg.norm(y-x)/np.sqrt(max(y.shape[0], y.shape[1])) #RMS
        if(name == 'classification'):
            def cross_entropy(yhat, y):
                epsilon=1e-12
                yhat = np.clip(yhat, epsilon, 1. - epsilon)
                n = yhat.shape[1]
                ce = -np.sum(y*np.log(yhat))/n
                return ce
            return cross_entropy
        else:
            print('unknown usage => regression')
            return lambda x, y: np.linalg.norm(y-x)/np.sqrt(max(y.shape[0], y.shape[1])) #RMS
