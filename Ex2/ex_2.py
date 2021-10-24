import numpy as np
from io import BytesIO
import sys


class multi_perceptron():
    def __init__(self,epochs = 86,rate=0.0083,input_size = 12, class_size = 3): # like given zero in kelet ? check
        self.epochs = epochs
        self.rate = rate
        self.w = np.random.uniform(-1,1,(class_size,input_size+1))

    def train(self,train_x,train_y):
        for i in range(self.epochs):
            # shuffle
            indices = np.arange(train_x.shape[0])
            np.random.shuffle(indices)
            train_x = train_x[indices]
            train_y = train_y[indices]
            for x_i, y_i in zip(train_x,train_y):
                #predict
                y_hat = np.argmax(np.dot(self.w,x_i))
                # update
                if y_i!= y_hat:
                    self.w[y_i, :] += self.rate*x_i
                    self.w[y_hat, :] -= self.rate*x_i
    # predict y after training
    def predict(self, x):
        y_hat = np.argmax(np.dot(self.w, x))
        return y_hat

class multi_pa():
    def __init__(self,epochs =10,input_size = 12,class_size =3):
        self.epochs=epochs
        self.input_size=input_size
        self.w = np.zeros((class_size,input_size+1))
    def train(self,train_x,train_y):
        for i in range(self.epochs):
            # shuffle
            indices = np.arange(train_x.shape[0])
            np.random.shuffle(indices)
            train_x = train_x[indices]
            train_y = train_y[indices]
            for x_i, y_i in zip(train_x, train_y):
                # predict
                w_copy = np.delete(self.w, y_i, axis=0)
                y_hat = np.argmax(np.dot(w_copy, x_i))
                if(y_hat>= y_i):
                    y_hat+=1
                # update
                if y_i != y_hat:
                    w_part= 1.0-np.dot(self.w[y_i,:],x_i)+np.dot(self.w[y_hat,:],x_i)
                    l = max(0.0,w_part)
                    t_base = 2.0*((np.linalg.norm(x_i)**2.0))
                    # case norma of x equals 0
                    if( t_base == 0.0):
                        t_base = 1.0
                    t = l/t_base
                    self.w[y_i, :] += t * x_i
                    self.w[y_hat, :] -= t * x_i
    # predict y after training
    def predict(self, x):
        y_hat = np.argmax(np.dot(self.w, x))
        return y_hat

class multi_KNN():
    def __init__(self,k = 8,input_size = 355, class_size =3):
        self.input_size = input_size
        self.class_size = class_size
        self.k = k

    def train(self,train_x,train_y,test_x):

        self.input_size = test_x.shape[0]
        results = np.zeros(self.input_size)
        for x_i in range(self.input_size):
            # find k closest to x
            distances = np.linalg.norm(test_x[x_i] - train_x, axis=1)
            closest = np.argsort(distances)
            closest_i = train_y[closest]
            # sort by index
            k_closest_i =[]
            if (self.k >= train_y.shape[0]):
                k_closest_i = closest_i
            else:
                k_closest_i = closest_i[:self.k]
            # predict
            vals, inverse, count = np.unique(k_closest_i, return_inverse=True, return_counts=True)
            idx_vals_repeated = np.where(count > 1)[0]
            vals_repeated = vals[idx_vals_repeated]
            # if there is a val that repeats
            if (idx_vals_repeated.size):
                if (vals_repeated.size != 1):
                    # take max
                    idx_max_repeated = np.where(count == (np.amax(count)))[0]
                    max_repeated = vals[idx_max_repeated]
                    if(max_repeated.size >1):
                        # take the lowest
                        results[x_i] = np.amin(max_repeated)
                    else:
                        results[x_i] = max_repeated
                else:
                    # take one that repeats
                    results[x_i] = vals_repeated
            else:
                # take the lowest
                results[x_i]=(np.amin(vals))
        return results

def main():
    # arguments
    train_x,train_y,test_x = sys.argv[1], sys.argv[2],sys.argv[3]
    w_val = b"0"
    r_val = b"1"
    x_data = open(train_x,'rb').read()
    x_data = x_data.replace(b"W",w_val).replace(b"R",r_val)
    x_train = np.genfromtxt(BytesIO(x_data),delimiter=",")
    y_train = np.genfromtxt((train_y),delimiter=",",dtype=np.int32)
    test_data = open(test_x,'rb').read()
    test_data = test_data.replace(b"W",w_val).replace(b"R",r_val)
    x_test = np.genfromtxt(BytesIO(test_data),delimiter=",")
    x_size = x_train.shape[0]
    test_size = x_test.shape[0]

    # normalization val into [0,1] range
    features_num = x_train.shape[1]
    for i in range(features_num):
        v = x_train[:, i]
        test_v = x_test[:, i]
        min = np.amin(v)
        max = np.amax(v)
        # prevent 0 division
        if (min == max):
            x_train[:, i] = 0
            x_test[:, i] = 0
        else:
            x_train[:, i] = (v - min) / (max - min)
            x_test[:, i] = (test_v - min) / (max - min)

    # knn prediction
    knn = multi_KNN(input_size=x_test.shape[0])
    knn_results = knn.train(x_train,y_train,x_test)
    # add bias column to test a train x
    x_train = np.append(x_train, np.ones((x_size, 1)), axis=1)
    x_test = np.append(x_test, np.ones((test_size, 1)), axis=1)
    # pa predication
    pa = multi_pa()
    pa.train(x_train,y_train)
    # perceptron prediction
    pr = multi_perceptron()
    pr.train(x_train,y_train)
    # print to screen
    for x in range(x_test.shape[0]):
        print(f"knn: {int(knn_results[x])}, perceptron: {pr.predict(x_test[x])}, pa: {pa.predict(x_test[x])}")



if __name__ == "__main__":
    main()