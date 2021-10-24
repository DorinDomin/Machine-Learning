import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets

class ModelA(nn.Module):
    def __init__(self,image_size):
        super(ModelA,self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size,100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50,10)
    def forward(self,x):
        x =x.view(-1,self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x))

class ModelB(nn.Module):
    def __init__(self,image_size):
        super(ModelB,self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size,100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50,10)
    def forward(self,x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x))
class ModelC(nn.Module):
    def __init__(self,image_size,dropout):
        super(ModelC,self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size,100)
        self.dpo = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(100,50)
        self.dp1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(50,10)
    def forward(self,x):
        x =x.view(-1,self.image_size)
        x = F.relu(self.fc0(x))
        # Apply dropout
        x = self.dpo(x)
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dp1(x)
        return F.log_softmax(self.fc2(x))
class ModelD(nn.Module):
    def __init__(self,image_size):
        super(ModelD,self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size,100)
        self.bch1 = nn.BatchNorm1d(num_features=100)
        self.fc1 = nn.Linear(100,50)
        self.bch2 = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        x =x.view(-1,self.image_size)
        x = F.relu(self.bch1(self.fc0(x)))
        x = F.relu(self.bch2(self.fc1(x)))
        return F.log_softmax(self.fc2(x))
class ModelE(nn.Module):
    def __init__(self,image_size):
        super(ModelE,self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size,128)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,10)
        self.fc5 = nn.Linear(10,10)
        self.fc6 = nn.Linear(10,10)

    def forward(self,x):
        x =x.view(-1,self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.log_softmax(self.fc6(x))

class ModelF(nn.Module):
    def __init__(self,image_size):
        super(ModelF,self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size,128)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,10)
        self.fc5 = nn.Linear(10,10)
        self.fc6 = nn.Linear(10,10)
    def forward(self,x):
        x =x.view(-1,self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return F.log_softmax(self.fc6(x))
# final network
class Network():
    def __init__(self,optimizer,model,dropout=-1.0,epochs=10,r = 0.175,batch_size = 64):
        self.model= model
        self.epochs = epochs
        self.rate = r
        self.batch_size = batch_size
        if(dropout != -1):
            self.net = self.model(28*28,dropout)
        else:
            # no dropout
            self.net = self.model(28*28)
        self.opt = optimizer(self.net.parameters(), lr=self.rate)

    def train(self,train_loader):
        self.net.train()
        for batch_idx ,(data,labels) in enumerate(train_loader):
            # clean grd
            self.opt.zero_grad()
            # forward for predication
            output = self.net.forward(data)
            loss = F.nll_loss(output, labels)
            loss.backward()
            self.opt.step()

    def validation(self, test_loader):
        self.model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                result = self.net.forward(data)
                # sum up
                loss+= F.nll_loss(result,target,size_average=False).item()
                # get the index of the max log-prob.
                pr = result.max(1,keepdim= True)[1]
                correct+= pr.eq(target.view_as(pr)).cpu().sum()
        # for debug
        test_loader_len = len(test_loader.dataset)
        loss/=test_loader_len
        print('validation result: avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(loss,correct,
              test_loader_len,100.*correct/test_loader_len))
        return correct
    # for checking
    def train_and_valid(self, train_loader, test_loader):
        best_correction = 0
        correct = 0
        for ep in range(self.epochs):
            self.train(train_loader)
            correct = self.validation(test_loader)
            if(correct > best_correction):
                best_correction = correct
        return best_correction

# find best vals for models: A,B,D,E,F
def find_best_model(training,valid):
    best_corr = 0
    correction = 0
    final_opt = ""
    best_model = ""
    best_rate = 0
    d =0
    best_d = 0
    valid_size = len(valid.dataset)
    opt = {"SGD":optim.SGD,"ADAM":optim.Adam,"RMSprop":optim.RMSprop,"AdaDelta":optim.Adadelta}
    models = {"A": ModelA,"B":ModelB,"C":ModelC,"D":ModelD,"E":ModelE,"F":ModelF}
    for m_name,m in models.items():
        # find h - params for each model
        optimizer,rate,d = find_best_optimizer(training,valid,m,m_name) # להחזירררר
        print(f"**************** testing model: {m_name} ****************")
        # simulate model and save results
        curr_net = Network(opt[optimizer],m,r= rate,dropout=d)
        correction = curr_net.train_and_valid(training,valid)
        print(f"[!!] finish checking model: {m_name} opt: {optimizer} rate: {rate} drop: {d} "
              f"correct: {(correction/valid_size)*100}%")
        # save best model
        if(correction > best_corr):
            best_corr = correction
            best_model = m_name
            final_opt = optimizer
            best_rate = rate
            best_d = d
    print(f"[***] best model: {best_model} optimizer: {final_opt} rate: {best_rate} best_dropout: {best_d} correct: {(best_corr/valid_size)*100}%")
def find_best_optimizer(training,valid,model,model_name):
    opt = {"SGD":optim.SGD,"ADAM":optim.Adam,"RMSprop":optim.RMSprop,"AdaDelta":optim.Adadelta}
    rate = 0
    best_rate = 0
    best_opt = ""
    correction = 0
    best_corr = 0
    best_drop = -1
    print(f"[!] finding h-params for model: {model_name}")
    if ((model_name == "A") or  (model_name == "D")):
        best_opt = "SGD"
        best_rate,best_corr = find_best_rate(training,valid,model,model_name,opt[best_opt],best_drop)
    elif (model_name == "C"):
        best_opt = "SGD"
        # find best dropout val
        for d in np.arange(0.05,1,0.05):
            rate, correction = find_best_rate(training, valid, model, model_name, opt[best_opt],d)
            if (correction > best_corr):
                best_corr = correction
                best_drop = d
                best_rate = rate
    elif (model_name == "B"):
        best_opt = "ADAM"
        best_rate,best_corr = find_best_rate(training,valid,model,model_name,opt[best_opt],best_drop)
    else:
        # E or F- find best optimizer
        for opt_name, f in opt.items():
            rate, correction = find_best_rate(training, valid, model, model_name, f,best_drop)
            if (correction > best_corr):
                best_opt = opt_name
                best_rate = rate
                best_corr = correction
    return (best_opt, best_rate,best_drop)

def find_best_rate(training,valid,model,model_name,optimizer,drop):
    best_rate = 0
    correction = 0
    best_corr = 0
    valid_size = len(valid.dataset)
    for r in np.arange(0.005,0.5,0.005):
        current_net = Network(optimizer,model,r=r,dropout=drop)
        current_net.train(training)
        correction = current_net.validation(valid)
        if (correction > best_corr):
            best_corr = correction
            best_rate = r
    print(f" model: {model_name} optimizer: {optimizer} best rate: {best_rate} drop: {drop} correction: { best_corr/valid_size}")
    return (best_rate,correction)

# given params
train_x, train_y, test_x = sys.argv[1], sys.argv[2], sys.argv[3]
transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
# for testing- 20:80
training_db = datasets.FashionMNIST("./data", train=True, download=True,transform=transforms)
# split given train into training and validation
train_part, valid_part = torch.utils.data.random_split(training_db, [round(len(training_db)*0.8),
    len(training_db)-round(len(training_db)*0.8)])
# load
train_loader = torch.utils.data.DataLoader(train_part,batch_size = 64,shuffle = True)
valid_loader =  torch.utils.data.DataLoader(valid_part,batch_size = 64,shuffle = True)
#test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST ("./data", train=False,
#                                                     transform=transforms),batch_size=64, shuffle=True)
# ************ for final submission************
#load test and normalize given test X
test_x = np.loadtxt(test_x) / 255
test_x = transforms(test_x).float()
final_n = Network(optimizer= optim.SGD,r=0.175,model=ModelD)
for ep in range(final_n.epochs):
    final_n.train(train_loader)
# print predictions
with open("test_y","w") as file:
    for t in test_x[0]:
        final_n.net.eval()
        with torch.no_grad():
            # predict
            p = final_n.net(t).argmax()
        file.write(str(int(p)))
        file.write("\n")