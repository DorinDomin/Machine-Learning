import torch
import numpy as np
import sys
import torch
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import os.path

import soundfile as sf
import librosa
import torch.utils.data as data

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = sf.read(path)
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)

        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.len = len(self.spects)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return spect, target

    def __len__(self):
        return self.len
# model-rnn probably
class Model(nn.Module):
    def __init__(self,dropout): #,dropout
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(7680, 30)
        self.fc11 = nn.Linear(30, 30)

        self.fc2 = nn.Linear(30, 30)
        self.fc21 = nn.Linear(30, 30)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc11(out)
        out = self.fc2(out)
        out = self.fc21(out)
        return  F.log_softmax(out,dim =1)
# final network
class Network():
    def __init__(self,model,optimizer,input_size=101*161,output_size =30,n_layers=1,hidden_dim=191,dropout=0.1,
                 epochs=10,r = 0.325,
                 batch_size = 10):
        self.model= model
        self.input_size = input_size
        self.epochs = epochs
        self.rate = r
        self.batch_size = batch_size
        self.net = self.model(dropout)
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
        # debug
        #print("Train Loss: {:.4f}".format(loss.item()))
    def validation(self, test_loader):
        self.net.eval()
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
        
        return correct
    # for checking
    def train_and_valid(self, train_loader, test_loader):
        best_correction = 0
        correct = 0
        for ep in range(self.epochs):
            #print('Epoch: {}/{}.............'.format(ep, self.epochs), end=' ')
            self.train(train_loader)
            correct = self.validation(test_loader)
            if(correct > best_correction):
                best_correction = correct
        return best_correction

def find_best_optimizer(training,valid,model,model_name):
    opt = {"AdaDelta": optim.Adadelta}
    rate = 0
    best_rate = 0
    best_rate_d = 0
    best_opt = ""
    correction = 0
    correction_m = 0
    correction_d = 0
    best_corr = 0
    best_drop = -1
    best_drop_d = -1
    valid_size = len(valid_loader)
    for opt_name, f in opt.items():
        correction_m = 0
        print(f"[!] finding rate for optimizer: {opt_name}")
        # find best dropout val
        rate, correction_d = find_best_rate(training, valid, model, model_name, f, -1.0)
        if (correction_d > correction_m):
            correction_m = correction_d
            best_drop_d = -1.0
            best_rate_d = rate
        print(f"**************** testing rate: {best_rate_d} ****************")
        # simulate model and save results
        curr_net = Network(optimizer=f, model=model, r=best_rate_d,dropout=best_drop_d)
        correction = curr_net.train_and_valid(training, valid)
        if (correction > best_corr):
            best_opt = opt_name
            best_rate = best_rate_d
            best_corr = correction
            best_drop = best_drop_d
        print(f"[!!] finish checking opt: {opt_name}  rate: {best_rate_d} drop: {best_drop_d} "
                  f"correct: {(correction/valid_size)*100}%")
    print(f"[***] best optimizer: {best_opt} rate: {best_rate} dropout: {best_drop} "
          f"correct: {100*(best_corr/valid_size)}%")

def find_best_rate(training,valid,model,model_name,optimizer,drop):
    best_rate = 0
    correction = 0
    best_corr = 0
    valid_size = len(valid.dataset)
    for r in np.arange(0.005,0.5,0.005):
        print(f"[*] checking rate: {r}")
        current_net = Network(optimizer=optimizer,model=model,r=r,dropout=drop)
        current_net.train(training)
        correction = current_net.validation(valid)
        if (correction > best_corr):
            best_corr = correction
            best_rate = r
    print(f" model: {model_name} optimizer: {optimizer} best rate: {best_rate} drop: {drop} correction: {100.*  (best_corr/valid_size)}")
    return (best_rate,correction)

train_set = GCommandLoader('C:\\Users\\דורין דומין\\Documents\\לימודים\\למידת מכונה\\ex5\\train')
valid_set = GCommandLoader('C:\\Users\\דורין דומין\\Documents\\לימודים\\למידת מכונה\\ex5\\valid')
test_set = GCommandLoader('C:\\Users\\דורין דומין\\Documents\\לימודים\\למידת מכונה\\ex5\\test')

# load
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=1, shuffle=True,
         pin_memory=True)
valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=1, shuffle=True,
         pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False,
         pin_memory=True)
check_net = Network(model=Model,optimizer= torch.optim.Adadelta,r= 0.315)
check_net.train_and_valid(train_loader,valid_loader)
# print predictions
preds = []
dict = {"0":"bed","1":"bird","2":"cat","3":"dog","4":"down","5":"eight","6":"five","7":"four","8":"go","9":"happy",
        "10":"house","11":"left","12":"marvin","13":"nine","14":"no","15":"off","16":"on","17":"one","18":"right",
        "19":"seven","20":"sheila","21":"six","22":"stop","23":"three","24":"tree","25":"two","26":"up","27":"wow",
        "28":"yes","29":"zero"}
i =0
check_net.net.eval()
with torch.no_grad():
    for data, target in test_loader:
        # predict
        output = check_net.net(data)
        p = output.argmax()
        cl = dict[str(int(p))]
        name = test_set.spects[i][0].rsplit('\\',1)[1]
        #print(name+","+cl)
        preds.append(str(name+","+cl))
        i+=1
preds = sorted(preds,key=lambda x:int(x.split('.')[0]))

with open("test_y","w") as file:
    for pr in preds:
        file.write(pr)
        file.write("\n")
