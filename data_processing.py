# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:18:27 2018

@author: lenovo
"""

import data_io_func
import torch.utils.data as Data
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch
#import datetime
import torch.nn.init as init
import torch.nn.functional as F
import pickle
#parameters setting--------------------------------------------
max_pep_seq_length=15
batch_size=32
epochs=15
motif_length=9
n_filters=20
learning_rate=0.01
dropout=0.5
n_hid=60#Linear
#W_INIT=uniform
#print("# w_init: " + str(W_INIT))
#COST_FUNCTION=squared_error
#SEED=-1
#update=sgd

#import dataset----------------------------------------------
X_pep_train1,X_mhc_train1,y_train1 = data_io_func.netcdf2pep('data/train1.bl.nc')
X_pep_train2,X_mhc_train2,y_train2 = data_io_func.netcdf2pep('data/train2.bl.nc')
X_pep_train3,X_mhc_train3,y_train3 = data_io_func.netcdf2pep('data/train3.bl.nc')
X_pep_train4,X_mhc_train4,y_train4 = data_io_func.netcdf2pep('data/train4.bl.nc')
X_pep_train5,X_mhc_train5,y_train5 = data_io_func.netcdf2pep('data/train5.bl.nc')

X_pep_e,X_mhc_e,y_e=data_io_func.netcdf2pep('data/evaluation.bl.nc')

X_pep_train=X_pep_train1#+X_pep_train2+X_pep_train3+X_pep_train4+X_pep_train5
X_mhc_train=X_mhc_train1#+X_mhc_train2+X_mhc_train3+X_mhc_train4+X_mhc_train5
y_train=y_train1#+y_train2+y_train3+y_train4+y_train5
del X_pep_train1,X_mhc_train1,y_train1
del X_pep_train2,X_mhc_train2,y_train2
del X_pep_train3,X_mhc_train3,y_train3,X_pep_train4,X_mhc_train4,y_train4
del X_pep_train5,X_mhc_train5,y_train5
#--------dev or val---------
X_pep_val1,X_mhc_val1,y_val1 = data_io_func.netcdf2pep('data/test1.bl.nc')
X_pep_val2,X_mhc_val2,y_val2 = data_io_func.netcdf2pep('data/test2.bl.nc')
X_pep_val3,X_mhc_val3,y_val3 = data_io_func.netcdf2pep('data/test3.bl.nc')
X_pep_val4,X_mhc_val4,y_val4 = data_io_func.netcdf2pep('data/test4.bl.nc')
X_pep_val5,X_mhc_val5,y_val5 = data_io_func.netcdf2pep('data/test5.bl.nc')

X_pep_val=X_pep_val1#+X_pep_val2+X_pep_val3+X_pep_val4+X_pep_val5
X_mhc_val=X_mhc_val1#+X_mhc_val2+X_mhc_val3+X_mhc_val4+X_mhc_val5
y_val=y_val1#+y_val2+y_val3+y_val4+y_val5

del X_pep_val1,X_mhc_val1,y_val1 ,X_pep_val2,X_mhc_val2,y_val2
del X_pep_val3,X_mhc_val3,y_val3
del X_pep_val4,X_mhc_val4,y_val4,X_pep_val5,X_mhc_val5,y_val5

# get MHC pseudo sequence length (assumes they all have the same length):
MHC_SEQ_LEN = X_mhc_train[0].shape[0]
# get target length:
T_LEN = 1#y_train[0].shape[0]

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

if MAX_PEP_SEQ_LEN == -1:
    # no length restraint -> find max length in dataset
    MAX_PEP_SEQ_LEN = max( len(max(X_pep_train, key=len)), len(max(X_pep_val, key=len)) )
else:
    # remove peptides with length longer than max peptide length:
    idx=[i for i,x in enumerate(X_pep_train) if len(x) > MAX_PEP_SEQ_LEN]

    X_pep_train = [i for j, i in enumerate(X_pep_train) if j not in idx]
    X_mhc_train = [i for j, i in enumerate(X_mhc_train) if j not in idx]
    y_train = [i for j, i in enumerate(y_train) if j not in idx]

    idx=[i for i,x in enumerate(X_pep_val) if len(x) > MAX_PEP_SEQ_LEN]

    X_pep_val = [i for j, i in enumerate(X_pep_val) if j not in idx]
    X_mhc_val = [i for j, i in enumerate(X_mhc_val) if j not in idx]
    y_val = [i for j, i in enumerate(y_val) if j not in idx]
    
    idx=[i for i,x in enumerate(X_pep_e) if len(x) > MAX_PEP_SEQ_LEN]

    X_pep_e = [i for j, i in enumerate(X_pep_e) if j not in idx]
    X_mhc_e = [i for j, i in enumerate(X_mhc_e) if j not in idx]
    y_e = [i for j, i in enumerate(y_e) if j not in idx]

# save sequences as np.ndarray instead of list of np.ndarrays:

X_pep_train = data_io_func.pad_seqs_T(X_pep_train, MAX_PEP_SEQ_LEN)
X_mhc_train = data_io_func.pad_seqs_T(X_mhc_train, MHC_SEQ_LEN)
x_train=np.dstack((X_pep_train, X_mhc_train))

X_pep_val = data_io_func.pad_seqs_T(X_pep_val, MAX_PEP_SEQ_LEN)
X_mhc_val = data_io_func.pad_seqs_T(X_mhc_val, MHC_SEQ_LEN)
x_val=np.dstack((X_pep_val,X_mhc_val))

X_pep_e = data_io_func.pad_seqs_T(X_pep_e, MAX_PEP_SEQ_LEN)
X_mhc_e = data_io_func.pad_seqs_T(X_mhc_e, MHC_SEQ_LEN)
x_e=np.dstack((X_pep_e, X_mhc_e))


N_FEATURES = X_pep_train.shape[1]#21

y_train = data_io_func.pad_seqs(y_train, T_LEN)
y_val = data_io_func.pad_seqs(y_val, T_LEN)
y_e = data_io_func.pad_seqs(y_e, T_LEN)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)#.long()

x_val = Variable(torch.from_numpy(x_val))
y_val = Variable(torch.from_numpy(y_val))
y_val = torch.squeeze(input=y_val, dim=1).cuda()
y_val=y_val.cuda()

x_e = torch.from_numpy(x_e)
y_e = torch.from_numpy(y_e)#.long()


'''
fw = open('data/Train_data_all.txt','wb')  
pickle.dump(x_train, fw, -1)
pickle.dump(y_train, fw,-1)  
fw.close()

fw = open('data/Dev_data_all.txt','wb')  
pickle.dump(x_val, fw, -1)
pickle.dump(y_val, fw,-1)  
fw.close()

'''
fw = open('data/Train_data1.txt','wb')  
pickle.dump(x_train, fw, -1)
pickle.dump(y_train, fw,-1)  
fw.close()

fw = open('data/Dev_data1.txt','wb')  
pickle.dump(x_val, fw, -1)
pickle.dump(y_val, fw,-1)  
fw.close()

fw = open('data/Eval_data.txt','wb')  
pickle.dump(x_e, fw, -1)
pickle.dump(y_e, fw,-1)  
fw.close()