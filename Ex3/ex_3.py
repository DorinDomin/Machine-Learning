import numpy as np
import sys
from scipy.special import softmax
# from tirgul
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
class multy_net():
    def __init__(self,input_size,neurons_num,epochs = 30,rate=0.02, class_size = 10,lightsness_len = 255):
        self.data_size = input_size
        self.neurons_num = neurons_num
        self.epochs = epochs
        self.rate = rate
        self.num_of_class = class_size
        self.lightness_limit = lightsness_len
        # for hidden layers
        np.random.seed(1)
        random_b = (np.sqrt(1.0 / 332))
        self.w_in = np.random.uniform(-random_b, random_b, (neurons_num, input_size))
        self.b_in = np.random.uniform(-0.0, 0.0, (neurons_num, 1))
        # for classification
        self.w_out = np.random.uniform(-random_b, random_b, (class_size, neurons_num))
        self.b_out = np.random.uniform(-0.0, 0.0, (class_size, 1))
        self.all_losses = []

    def train(self,train_x,train_y):
        for i in range(self.epochs):
            sum_loss = 0
            # shuffle
            indices = np.arange(train_x.shape[0])
            np.random.shuffle(indices)
            train_x = train_x[indices]
            train_y = train_y[indices]
            # loop over the training set
            for x_i, y_i in zip(train_x,train_y):
                # Forward the input instance through the network
                y_hat = self.forwardPropagation(x_i)
                # Calculate the loss
                l = -np.log(y_hat[y_i]) # ככה התכוונו לממש את זה ?
                sum_loss = sum_loss+ l
                # if predication is correct- no need to update
                if (l != 0.0):
                    # Compute the gradients w.r.t all the parameters (backpropagation)
                    self.backPropagation(x_i,y_i)
            self.all_losses.append(float(sum_loss/train_x.shape[0]))
            # print(f"epoch: {i} loss: {sum_loss/train_x.shape[0]}") #למחוק
            # self.validation(train_x,train_y)
    def predict(self,val):
        # run throw net
        y_tag = self.forwardPropagation(val)
        # take biggest prob. of classes
        return (np.argmax(y_tag))
    def backPropagation(self,x,y):
        x = np.array(x)
        # softmax d
        dL_dytag = self.h2
        dL_dytag[y] = dL_dytag[y] -1
        # Gradient w.r.t input
        d_b = np.reshape(dL_dytag, (self.num_of_class, 1))
        # correct way of multiplication (neurons_num * 1) into (1 * neurons_num)
        w_tag = np.reshape(np.transpose(self.h1),(1,self.neurons_num))
        #  dL/dz2 * dz2/dw2 (10 * 1)*(1 * neurons_num)
        d_l_input = np.dot(d_b,w_tag)
        # Gradient w.r.t parameters
        d_h = np.dot(np.transpose(self.w_out),dL_dytag)
        # segmoid d
        tr = (self.h1 * (1.0-self.h1))
        d_h *= (tr)
        # make sure you multiply (neurons_num * 1) over * (which is one picture of 28*28)
        dW = np.dot(np.reshape(d_h, (self.neurons_num, 1)), np.reshape(x,(1,28*28))) # לבדוק האם צריך לסדר מחדש או להפוך כמו שעשיתי ?
        db1 = d_h
        # Update the parameters using GD/SGD (from lecture notes 8)
        self.w_out += -self.rate*d_l_input
        self.b_out+= -self.rate*d_b
        self.w_in+= -self.rate* dW
        self.b_in += -self.rate*db1

    def forwardPropagation(self,val):
        # forward throw the net (only one layer)
        self.z1= np.dot(self.w_in,np.array([val]).T)+ self.b_in
        # activation function
        self.h1 = sigmoid(self.z1)
        # last forward- into the classification layer
        self.z2 = np.dot(self.w_out,self.h1)+ self.b_out
        # softmax activation over the output layer
        self.h2 = softmax(self.z2)
        return self.h2

def main():
    # load data
    train_x,train_y,test_x = sys.argv[1], sys.argv[2],sys.argv[3]
    x_train = np.loadtxt(train_x)
    y_train = np.loadtxt(train_y,dtype=np.int32)
    x_test = np.loadtxt(test_x)
    # normalization
    x_train/= 255
    x_test/=255
    # model activation
    net = multy_net(x_train.shape[1],105,epochs=30,rate=0.02)
    net.train(x_train,y_train)
    output = open("test_y", "w+")
    # test_x predication
    for x in x_test:
        pr = net.predict(x)
        output.write(f"{int(pr)}\n")
    output.close()
if __name__ == "__main__":
    main()
