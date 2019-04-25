import math, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Bottleneck(nn.Module):
  def __init__(self, nChannels, growthRate):
    super(Bottleneck, self).__init__()
    interChannels = 4*growthRate
    self.bn1 = nn.BatchNorm2d(nChannels)
    self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=(1,3), bias=False)
    self.bn2 = nn.BatchNorm2d(interChannels)
    self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=(3,21), padding=1, bias=False)

  def forward(self, x):
    
    out=self.bn1(x)
    out=F.relu(out)#
    out=self.conv1(out)
    
    #out = self.conv1(F.relu(self.bn1(x)))
    out=self.bn2(x)
    out=F.relu(out)#([64, 12, 15, 3])
    out=self.conv2(out)
    #out = self.conv2(F.relu(self.bn2(out)))
    out = torch.cat((x, out), 1)
    return out

class SingleLayer(nn.Module):
  def __init__(self, nChannels, growthRate):
    super(SingleLayer, self).__init__()
    self.bn1 = nn.BatchNorm2d(nChannels)
    self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=(3,1), padding=(1,0), bias=False)
    #self.mlp=nn.Linear()
    self.Linear1=nn.Linear(InChannels*15,1)
    #self.Linear2=nn.Linear(growthRate*15,1)
    self.Linear2 = nn.Parameter(torch.randn(growthRate*15,1).t())

    self.Linear11=nn.Linear(InChannels*34,1)
    #self.Linear22=nn.Linear(growthRate*34,1)
    self.Linear22 = nn.Parameter(torch.randn(growthRate*34,1).t())
    self.Linear111=nn.Linear(InChannels*17,1)
    #self.Linear222=nn.Linear(growthRate*17,1)
    self.Linear222 = nn.Parameter(torch.randn(growthRate*17,1).t())
    self.Linear51=nn.Linear(InChannels*5,1)
    #self.Linear52=nn.Linear(growthRate*5,1)
    self.Linear52 = nn.Parameter(torch.randn(growthRate*5,1).t())
    self.Linear61=nn.Linear(InChannels*11,1)
    #self.Linear62=nn.Linear(growthRate*11,1)
    self.Linear62 = nn.Parameter(torch.randn(growthRate*11,1).t())
  def forward(self, x):
    #print('x', x.shape)
    if x.shape[1]==InChannels:
        #print(x.shape)
        out = self.conv1(F.relu(self.bn1(x),inplace=True))#(x=[64, 12, 15, 3],out=[64, 6, 15, 3])
        #print('conv(x)',out.shape)
        out = torch.cat((x,out), 1)#([64, 18, 15, 3])
        del x
        return out
    else:
        n=int((x.shape[1]-InChannels)/growth)
        x1=x[:,:InChannels,:,:]
        x2=x[:,InChannels:,:,:]
        long=x1.shape[2]
        #xxx=x1
        x11=torch.mean(x1,dim=0)
        
        if long==15:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=0)
            x_outt=x_outt.view(1,growth*15)
            x11=x11.view(1,InChannels*15)
            weight=torch.sigmoid(F.linear(x_outt,self.Linear2)+self.Linear1(x11))
            x1=weight*x1
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=0)
                #print(xx_temp.shape)
                xx_temp=xx_temp.view(1,growth*15)
                
                weight=torch.sigmoid(F.linear(xx_temp,self.Linear2)+F.linear(x_outt,self.Linear2))
                xx=weight*xx
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        elif long==34:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=0)
            x_outt=x_outt.view(1,growth*34)
            x11=x11.view(1,InChannels*34)
            weight=torch.sigmoid(F.linear(x_outt,self.Linear22)+self.Linear11(x11))
            x1=weight*x1
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=0)
                #print(xx_temp.shape)
                xx_temp=xx_temp.view(1,growth*34)
                
                weight=torch.sigmoid(F.linear(xx_temp,self.Linear22)+F.linear(x_outt,self.Linear22))
                xx=weight*xx
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        elif long==5:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=0)
            x_outt=x_outt.view(1,growth*5)
            x11=x11.view(1,InChannels*5)
            weight=torch.sigmoid(F.linear(x_outt,self.Linear52)+self.Linear51(x11))
            x1=weight*x1
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=0)
                #print(xx_temp.shape)
                xx_temp=xx_temp.view(1,growth*5)
                
                weight=torch.sigmoid(F.linear(xx_temp,self.Linear52)+F.linear(x_outt,self.Linear52))
                xx=weight*xx
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        elif long==11:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=0)
            x_outt=x_outt.view(1,growth*11)
            x11=x11.view(1,InChannels*11)
            weight=torch.sigmoid(F.linear(x_outt,self.Linear62)+self.Linear61(x11))
            x1=weight*x1
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=0)
                #print(xx_temp.shape)
                xx_temp=xx_temp.view(1,growth*11)
                
                weight=torch.sigmoid(F.linear(xx_temp,self.Linear62)+F.linear(x_outt,self.Linear62))
                xx=weight*xx
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        else:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=0)
            x_outt=x_outt.view(1,growth*17)
            x11=x11.view(1,InChannels*17)
            weight=torch.sigmoid(F.linear(x_outt,self.Linear222)+self.Linear111(x11))
            x1=weight*x1
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=0)
                #print(xx_temp.shape)
                xx_temp=xx_temp.view(1,growth*17)
                
                weight=torch.sigmoid(F.linear(xx_temp,self.Linear222)+F.linear(x_outt,self.Linear222))
                xx=weight*xx
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        xxx=torch.cat((xxx,x_out),1)
        #print(xxx)
        #xxx=torch.stack(xxx)
        out = self.conv1(F.relu(self.bn1(xxx),inplace=True))#(x=[64, 12, 15, 3],out=[64, 6, 15, 3])
        #print('conv(x)',out.shape)
        out = torch.cat((x,out), 1)#([64, 18, 15, 3])
        del x,x1,x11,xxx,weight,x_out,x_outt
        return out

class Transition(nn.Module):
  def __init__(self, nChannels, nOutChannels):
    super(Transition, self).__init__()
    self.bn1 = nn.BatchNorm2d(nChannels)
    self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=(1,1), bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x),inplace=True))#[64, 36, 15, 3]
    #print('aaaa!!!!!!!!!!!!!!!!aaa',out.shape)
    # out = F.avg_pool2d(out, 2)
    return out

class DenseNet(nn.Module):
  def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
    super(DenseNet, self).__init__()

    if bottleneck:  nDenseBlocks = int( (depth-4) / 6 )
    else         :  nDenseBlocks = int( (depth-4) / 3 )#多少个single layer是根据depth来算的。
    global growth
    growth=growthRate
    nChannels = 2*growthRate
    self.conv1 = nn.Conv2d(1, nChannels, kernel_size=(3,21), padding=(1,0), bias=False)
    global InChannels
    InChannels=nChannels
    self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate
    nOutChannels = int(math.floor(nChannels*reduction))
    self.trans1 = Transition(nChannels, nOutChannels)
    
    nChannels = nOutChannels
    self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate
    nOutChannels = int(math.floor(nChannels*reduction))
    self.trans2 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate

    self.bn1 = nn.BatchNorm2d(nChannels)
    self.fc = nn.Linear(nChannels, nClasses)
    #self.fc2=nn.Linear(nChannels,7)
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
    layers = []
    for i in range(int(nDenseBlocks)):
      if bottleneck:
        layers.append(Bottleneck(nChannels, growthRate))
      else:
        layers.append(SingleLayer(nChannels,growthRate))
      nChannels += growthRate
    return nn.Sequential(*layers)

  def forward(self, x):
    pep=x[:,:,:15]#[batch_size, 21, 15]
    mhc=x[:,:,15:]#[batch_size, 21, 34]
    pep=pep.permute(0,2,1)
    mhc=mhc.permute(0,2,1)
    
    pep = torch.unsqueeze(input=pep, dim=1)
    pep=pep.type(torch.cuda.FloatTensor)
    out_pep=self.conv1(pep)#([64, 12, 13, 1])
    del pep
    
    out_pep = self.trans1(self.dense1(out_pep))
    out_pep = F.max_pool2d(out_pep,(3,1))#([64, 36, 5, 1])
    out_pep = self.trans2(self.dense2(out_pep))
    out_pep = self.dense3(out_pep)#[64, 108, 5, 1]
    mhc = torch.unsqueeze(input=mhc, dim=1)
    mhc=mhc.type(torch.cuda.FloatTensor)
    out_m=self.conv1(mhc)
    del mhc
    
    out_m=self.trans1(self.dense1(out_m))
    #out_m = F.avg_pool2d(out_m, 2)
    out_m = F.max_pool2d(out_m, (3,1))#11
    out_m = self.trans2(self.dense2(out_m))
    out_m = F.max_pool2d(out_m, (2,1))#([64, 48, 5, 1])
    #out_m = F.max_pool2d(out_m, (5,1), stride=1,padding=0)
    #out_all=torch.cat((out_p,out_m), 1)
    out_m = self.dense3(out_m)#[64, 108, 17, 1]
    #print('pep',out_pep.shape)
    #print('mhc',out_m.shape)
    #out_m=self.fc2(out_m)
    
    out_all=np.sum((out_pep,out_m),0)
    del out_pep
    del out_m
    #out_all=torch.cat((out_p,out_m), 1)
    out_all=self.bn1(out_all)
   
    out_all=F.relu(out_all,inplace=True)
    out_all=F.max_pool2d(out_all,(5,1))
    out_all=torch.squeeze(out_all)
    #print('out_all',out_all.shape)
    #out_all = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out_all),inplace=True), 2))
    out_all = self.fc(out_all)
    out=torch.sigmoid(out_all)
    return out

def densenet100_12(num_classes=1):
  model = DenseNet(6, 36, 0.5, num_classes, False)
  return model
