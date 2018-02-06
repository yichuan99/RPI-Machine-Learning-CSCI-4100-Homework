import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import classify as cls
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(np.negative(x)))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

def identity(x):
    return x

def identity_deriv(x):
    return 1

class NeuralNetwork:
    def __init__(self, layers_node_counts, activation, activation_final):
        ##middle layer activation
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv        
        elif activation == 'identity':
            self.activation = identity
            self.activation_deriv = identity_deriv
        
        ##final layer activation
        if activation_final == 'logistic':
            self.activation_final = logistic
            self.activation_final_deriv = logistic_derivative
        elif activation_final == 'tanh':
            self.activation_final = tanh
            self.activation_final_deriv = tanh_deriv        
        elif activation_final == 'identity':
            self.activation_final = identity
            self.activation_final_deriv = identity_deriv
        elif activation_final == 'sign':
            self.activation_final = np.sign
            self.activation_final_deriv = 0
        
        ##store of values for every edge
        ##the bias parameter is included as the first node in that layer
        self.weights=[]
        self.biases=[]
        for i in range(len(layers_node_counts)-1):
            w = np.full((layers_node_counts[i],layers_node_counts[i+1]),0.25)
            self.weights.append(w)
            self.biases.append(np.full(layers_node_counts[i+1], 0.25))
        
        ##store of values for every node
        self.layers=[]
        for i in range(len(layers_node_counts)):
            layer = np.zeros(layers_node_counts[i])
            self.layers.append(layer)
            
        self.output = 0
    
    ##take in data_point and return the final layer activation values
    ##the data dimension should be consistent with the first input layer dimension 
    ##(the layer before the first hidden layer)
    def FrontProp(self, data_point):
        x,y = data_point
        
        ##pump input values into the first layer (the one before the first hidden layer)
        for i in range(len(self.layers[0])):
            self.layers[0][i]=x[i]
        
        ##propagation
        for i in range(len(self.weights)):
            w = self.weights[i]
            input_layer = self.layers[i]
            b = self.biases[i]
            next_layer = np.matmul(input_layer, w)+b    
            if(i==len(self.weights)-1):##final layer
                next_layer =  self.activation_final(next_layer)
            else: next_layer =  self.activation(next_layer)##middle layer
            self.layers[i+1]=next_layer
            
        answer = copy.deepcopy(self.layers[len(self.layers)-1])
        #print self.layers
        #print answer
        return answer
        
    def Err(self, hx, y):
        ##hx should be a 1-d array with only one element
        y_=hx[0]
        return 0.25*(y_-y)**2
    
    def Err_deriv(self, hx, y):
        ##hx should be a 1-d array with only one element
        y_=hx[0]
        return 0.25*(y_-y)*2
   
    def GradientFrontPropApprox(self, data_point, rate):
        x,y = data_point
        ##front propagation output, hx should be a 1-d array with only one element
        hx = self.FrontProp(data_point)
        err = self.Err(hx,y)
        
        ##nudge each weight to get its approximate gradient
        ##store the gradients in a structure similar to self.weights
        weight_gradients = []
        for i in range(len(self.weights)): 
            ##self.weights[i] is the 2-D weight matrix for ith layer of neurons
            
            weight_gradients_matrix = []
            for j in range(len(self.weights[i])):
                ##self.weights[i][j] is the jth row of matrix
                w_gradients_n_row = []
                
                ##it represents the group of nth weights for every neuron in current layer
                for k in range(len(self.weights[i][j])):
                    ##nudge this weight
                    self.weights[i][j][k]+=rate
                    ##get the output for nudged neural network
                    hx_nudge = self.FrontProp(data_point)
                    ##error cululation (hx_nudge should be a 1-D array with only one element)
                    err_nudged = self.Err(hx_nudge, y)
                    ##restore the previous weight
                    self.weights[i][j][k]-=rate
                    ##store the gradient value (front propagation approximation method)
                    w_gradients_n_row.append(float(err_nudged-err)/rate)
                ##put the gradient row into gradient matrix
                weight_gradients_matrix.append(w_gradients_n_row)
            ##put the gradient matrix for corresponding weight matrix into storage
            weight_gradients.append(weight_gradients_matrix)
            
        print weight_gradients
        print self.weights
        
        
        ##nudge each bias to get its approximate gradient
        ##store the gradients in a structure similar to self.biases        
        bias_gradients = []
        for i in range(len(self.biases)):
            bias_vector_gradients = []            
            
            for j in range(len(self.biases[i])):
                ##nudge this bias
                self.biases[i][j]+=rate
                ##get the output for nudged neural network
                hx_nudge = self.FrontProp(data_point)
                ##error cululation (hx_nudge should be a 1-D array with only one element)
                err_nudged = self.Err(hx_nudge, y)   
                ##restore original bias
                self.biases[i][j]-=rate
                ##store the gradient value (front propagation approximation method)
                bias_vector_gradients.append(float(err_nudged-err)/rate)                
            
            bias_gradients.append(bias_vector_gradients)
        
        print bias_gradients
        print self.biases                
    
    ##Back Propagation
    def BackProp(self, data_point):
        self.FrontProp(data_point)
        weight_gradients = []
        sensitivity_vectors = []
        
        for i in reversed(range(len(weights))):
            continue

##GG...
def DrawLinearContour(data, xmin, xmax, ymin, ymax):
    start_time = time.time()
            
    xlist = np.linspace(xmin, xmax, 250)
    ylist = np.linspace(ymin, ymax, 250)
    X, Y = np.meshgrid(xlist, ylist)
    Z=np.sign(X+2*Y+2.23)

    #plt.figure()
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.title('NeuralNet Plot')
    plt.xlabel('X1')
    plt.ylabel('X2')
    cls.draw_data(data)
    print("--- %s seconds ---" % (time.time() - start_time))   
    plt.show()
    
##GG again...
def PlotError():
    x = range(2*10**6)
    y = [0.2*2**(float(-x_)/10000) for x_ in x]
    plt.plot(x,y, "g.")
    plt.show()

##GG...
def DrawGGContour(data, xmin, xmax, ymin, ymax):
    
    xlist = np.linspace(xmin, xmax, 100)
    ylist = xlist**3
    plt.plot(xlist, ylist, "g-")
    x_1 = [0, 0.00001]
    y_1 = [ymin, ymax]
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.plot(x_1, y_1, "b-")
    cls.draw_data(data)
    plt.show()



DrawGGContour([([-1,0],-1),([1,0],1)], -1.5, 1.5, -1.5, 1.5)
'''
layers_node_counts = [2,2,1]
activation = 'tanh'
activation_final = 'identity'
my_net = NeuralNetwork(layers_node_counts, activation, activation_final)
data_point = [1,1],1
'''
#my_net.FrontProp(data_point)
#print my_net.layers
#my_net.GradientFrontPropApprox(data_point, 0.0001)
#PlotError()

