# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 18:18:25 2018

@author: lena
"""
#import data_io_func
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch
#import datetime
import torch.nn.init as init
import torch.nn.functional as F
import pickle
from sklearn.metrics import roc_auc_score
#parameters setting--------------------------------------------
max_pep_seq_length=15
batch_size=64
epochs=15
motif_length=9
n_filters=20
learning_rate=0.001
dropout=0.5
n_hid=60#Linear
#W_INIT=uniform
#print("# w_init: " + str(W_INIT))
#COST_FUNCTION=squared_error
#SEED=-1
#update=sgd
fr = open('data/Train_data1.txt', 'rb')
x_train = pickle.load(fr)#[196992, 21, 49]
y_train = pickle.load(fr)

fr = open('data/Dev_data1.txt', 'rb')
x_val = pickle.load(fr)#[49248, 21, 49]
y_val = pickle.load(fr)

# get MHC pseudo sequence length (assumes they all have the same length):
MHC_SEQ_LEN = 34#X_mhc_train[0].shape[0]
N_FEATURES = 21#X_pep_train.shape[1]

# get target length:
T_LEN = y_train[0].shape[0]

N_SEQS_VAL = y_val[0].shape[0]

MAX_PEP_SEQ_LEN=int(max_pep_seq_length)
print("# max. peptide sequence length: " + str(MAX_PEP_SEQ_LEN))
      
MOTIF_LEN=int(motif_length)
print("# mmotif length: " + str(MOTIF_LEN))
      
BATCH_SIZE=int(batch_size)
print("# batch size: " + str(BATCH_SIZE))
      
EPOCHS=range(1, int(epochs)+1)
print("# number of training epochs: " + str(epochs))
      
N_FILTERS=int(n_filters)
print("# number of convolutional filters: " + str(N_FILTERS))
      
LEARNING_RATE=float(learning_rate)
print("# learning rate: " + str(LEARNING_RATE))

DROPOUT=float(dropout)
print("# dropout: " + str(DROPOUT))

N_HID=int(n_hid)
print("# number of hidden units: " + str(N_HID))

torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
print("data process finished")

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.uniform_(m.weight,-0.05,0.05)
        m.bias.data.fill_(0.01)
        print('weights initialed with uniform')
    if isinstance(m, nn.Linear):
        init.uniform_(m.weight,-0.05,0.05)
        m.bias.data.fill_(0.01)
        print('weights initialed with uniform')
        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=N_FILTERS,#20
                kernel_size=(21,9),
                stride=1,
                padding=0
            ).cuda(),#[batch_size, n_channels, 1, 7]---7=15-9+1
            nn.BatchNorm2d(N_FILTERS, 1).cuda(),
            nn.ReLU().cuda(),
            nn.MaxPool2d(
                    kernel_size=(1,7),
                    padding=0,
                    stride=(1, 1)
                        ).cuda(),
        ).cuda()
        self.dp = nn.Dropout(p=DROPOUT).cuda()
        self.dense_p = nn.Linear(N_FILTERS,N_HID).cuda()
        self.dense_m = nn.Linear(N_FEATURES*MHC_SEQ_LEN,N_HID).cuda()
        self.dense_all = nn.Linear(2*N_HID,1).cuda()
        # torch.cat((x,y), 0)#x轴合并
        self.sgmoid = nn.Sigmoid().cuda()
        self.relu = nn.ReLU().cuda()
        #init.uniform(self.cnn.all_weights[0][0], gain=1)
        self.batch = nn.BatchNorm1d(N_HID*2).cuda()
        #init.xavier_normal(nn.Conv1d.weight)
        self.dropout=0.5
    def forward(self, x):
        pep=x[:,:,:15]#[batch_size, 21, 15]
        mhc=x[:,:,15:]#[batch_size, 21, 34]
        #pep=pep.view(:,1,:,:)
        #pep = torch.Tensor(pep).long()
        pep = torch.unsqueeze(input=pep, dim=1).cuda()
        out_pep=self.conv(pep)#([batch_size, 20, 1, 1])
        out_p=torch.squeeze(input=out_pep,dim=3).cuda()
        out_p=torch.squeeze(input=out_p,dim=2).cuda()#(batch_size, n_filters)
        #out_p= F.max_pool2d(out_pep)
        #size= len(out_p)
        mhc=mhc.contiguous().view(mhc.size(0), -1)
        mhc=mhc.cuda()
        out_m=self.dense_m(mhc)
        out_p=self.dense_p(out_p)
        out_all=torch.cat((out_p,out_m), 1)
        #out_all=np.sum((out_p,out_m),0)#可能需要改改改！！！
        #out_all=np.hstack((out_p,out_m)).detach()
        #out_all=torch.stack([out_p,out_m],dim=1)
        out_all=self.batch(out_all)
        out_all=self.relu(out_all)
        out_all = F.dropout(out_all, self.dropout, training=self.training)
        out=self.dense_all(out_all)
        out=self.sgmoid(out)
        return out
            
            #torch.cat((x,y), 0)
model = CNN()
model.apply(weights_init) 
#model.load_state_dict(torch.load('153851model.pkl'))
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8)
loss_func = nn.MSELoss()#默认值为各个MSE的平均值
print(model)
#f = open('loss_record.txt', 'w+')
best=0
y_binary = np.where(y_val>=0.42562, 1,0)
def get_test():
    global best
    model.eval()
    print('start validation')
    test_output = model(Variable(x_val)) 
    loss = loss_func(test_output, y_val)
    test_output=test_output.cpu().data.numpy()
    auc = roc_auc_score(y_binary.flatten(), test_output.flatten())
    #L = len(x_val)
    val_loss=float(loss)
    print('validation_error: ', val_loss) 
    print('validation_auc: ', auc)
    f.write(str(val_loss) + 'AUC:' +str(auc)+'\n')
    f.flush()
    if val_loss < best:
        torch.save(model, "CNN"+str(val_loss)+".pth")
        best = val_loss
    #f.write(str(float(right / L) * 100) + '\n')
    #f.flush()
    model.train()
    return test_output

f = open('validation_loss_record.txt', 'w+')
for e in EPOCHS:
    # shuffle training examples and iterate through minbatches:
    for index, (batch_x, batch_y) in enumerate(loader):
        right = 0
        if index == 0:
            get_test()
            loss_sum = 0
        #Size = len(batch_x)
        batch_x = Variable(batch_x)
        #   one hot to scalar
        batch_y = batch_y.cuda()
        batch_y = torch.squeeze(input=batch_y, dim=1)
        batch_y = Variable(batch_y.float())
        output = model(batch_x)
        optimizer.zero_grad()
        output = output.cuda()
        loss = loss_func(output, batch_y)
        loss.backward()
        #predict = output.cpu().numpy().tolist().detach()
        #label = batch_y.cpu().numpy().tolist().detach()
        optimizer.step()
        #loss_sum += float(loss)
        if index % 100 == 0:
            print("batch", index, "/ "+str(len(loader))+": ",  "\ttrain_error: ", float(loss))
    print('epoch: ', e, 'has been finish')
