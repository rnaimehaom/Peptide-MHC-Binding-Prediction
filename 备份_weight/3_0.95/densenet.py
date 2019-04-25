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
    self.bn1 = nn.BatchNorm2d(nChannels).cuda()
    self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=(3,1), padding=(1,0), bias=False).cuda()
    self.fc_w15 = nn.Parameter(torch.randn(15,1).t(),requires_grad=True).cuda()
    self.bias = nn.Parameter(torch.randn(1),requires_grad=True).cuda()
    self.fc_w34 = nn.Parameter(torch.randn(34,1).t(),requires_grad=True).cuda()
    self.fc_w5 = nn.Parameter(torch.randn(5,1).t(),requires_grad=True).cuda()
    self.fc_w11= nn.Parameter(torch.randn(11,1).t(),requires_grad=True).cuda()
    self.fc_w17 = nn.Parameter(torch.randn(17,1).t(),requires_grad=True).cuda()
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
        x1=x[:,:InChannels,:,:]#([64, 12, 15, 1])
        x2=x[:,InChannels:,:,:]
        #print('InChannels',InChannels)
        #xxx=x1
        batch_size=x1.shape[0]
        x11=torch.mean(x1,dim=1)
        long=x11.shape[1]#([64, 15, 1])
        x11 = torch.squeeze(input=x11)
        if long==15:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=1)
            #x_outt=x_outt.view(1,15)
            #x11=x11.view(1,15)
            x_outt = torch.squeeze(input=x_outt)
            
            weight=torch.sigmoid(F.linear(x_outt,self.fc_w15,self.bias)+F.linear(x11,self.fc_w15,self.bias))
            #x1=weight.t()*x1
            x1=x1.detach()
            #print(x1.shape)
            #print(weight.shape)
            #weight=
            #for i in range(0,x1.shape[0]):
                #if weight[i]<0.2:
                 #   weight[i]=0
                #x1[i,:,:,:]=weight[i]*x1[i,:,:,:].clone()
            #print(x1.shape)
            #xt=weight.t().mul(x1)
            #x1=torch.dot(weight, x1)
            #weight = torch.squeeze(input=weight)
            #batch_size=weight.shape[0]
            weight=weight.clone().expand(batch_size,12)
            weight=weight.repeat(1,1,15)
            weight=weight.view(batch_size,12,15,1)
            weight=weight*x1.clone()
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=1)
                #print(xx_temp.shape)
                xx_temp = torch.squeeze(input=xx_temp)
                weight=torch.sigmoid(F.linear(xx_temp,self.fc_w15,self.bias )+F.linear(x_outt,self.fc_w15,self.bias ))
                xx=xx.detach()
                w1=weight.clone().expand(batch_size,growth)
                w1=w1.repeat(1,1,15)
                w1=w1.view(batch_size,growth,15,1)
                xx=w1*xx.clone()
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        elif long==34:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=1)
            x_outt = torch.squeeze(input=x_outt)
            weight=torch.sigmoid(F.linear(x_outt,self.fc_w34,self.bias )+F.linear(x11,self.fc_w34,self.bias ))
            x1=x1.detach()
            w1=weight.clone().expand(batch_size,12)
            w1=w1.repeat(1,1,34)
            w1=w1.view(batch_size,12,34,1)
            x1=w1*x1.clone()
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=1)
                #print(xx_temp.shape)
                xx_temp = torch.squeeze(input=xx_temp)
                weight=torch.sigmoid(F.linear(xx_temp,self.fc_w34,self.bias )+F.linear(x_outt,self.fc_w34,self.bias))
                xx=xx.detach()
                w1=weight.clone().expand(batch_size,growth)
                w1=w1.repeat(1,1,34)
                w1=w1.view(batch_size,growth,34,1)
                xx=w1*xx.clone()
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        elif long==5:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=1)
            x_outt = torch.squeeze(input=x_outt)
            weight=torch.sigmoid(F.linear(x_outt,self.fc_w5,self.bias)+F.linear(x11,self.fc_w5,self.bias))
            x1=x1.detach()
            w1=weight.clone().expand(batch_size,12)
            w1=w1.repeat(1,1,5)
            w1=w1.view(batch_size,12,5,1)
            x1=w1*x1.clone()
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=1)
                #print(xx_temp.shape)
                xx_temp = torch.squeeze(input=xx_temp)
                weight=torch.sigmoid(F.linear(xx_temp,self.fc_w5,self.bias)+F.linear(x_outt,self.fc_w5,self.bias))
                xx=xx.detach()
                w1=weight.clone().expand(batch_size,growth)
                w1=w1.repeat(1,1,5)
                w1=w1.view(batch_size,growth,5,1)
                xx=w1*xx.clone()
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        elif long==11:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=1)
            x_outt = torch.squeeze(input=x_outt)
            weight=torch.sigmoid(F.linear(x_outt,self.fc_w11,self.bias)+F.linear(x11,self.fc_w11,self.bias))
            x1=x1.detach()
            w1=weight.clone().expand(batch_size,12)
            w1=w1.repeat(1,1,11)
            w1=w1.view(batch_size,12,11,1)
            x1=w1*x1.clone()
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=1)
                #print(xx_temp.shape)
                xx_temp = torch.squeeze(input=xx_temp)
                weight=torch.sigmoid(F.linear(xx_temp,self.fc_w11,self.bias)+F.linear(x_outt,self.fc_w11,self.bias))
                xx=xx.detach()
                w1=weight.clone().expand(batch_size,growth)
                w1=w1.repeat(1,1,11)
                w1=w1.view(batch_size,growth,11,1)
                xx=w1*xx.clone()
                xxx=torch.cat((xxx,xx),1)
                i=i+1
                
        elif long==17:
            x_out=x2[:,-growth:,:,:]
            x_outt=torch.mean(x_out,dim=1)
            x_outt = torch.squeeze(input=x_outt)
            weight=torch.sigmoid(F.linear(x_outt,self.fc_w17,self.bias)+F.linear(x11,self.fc_w17,self.bias))
            x1=x1.detach()
            w1=weight.clone().expand(batch_size,12)
            w1=w1.repeat(1,1,17)
            w1=w1.view(batch_size,12,17,1)
            x1=w1*x1.clone()
            xxx=x1
            for i in range(n-2,-1,-1):
                #xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx=x2[:,(i*growth):(i+1)*growth,:,:]
                xx_temp=torch.mean(xx,dim=1)
                #print(xx_temp.shape)
                xx_temp = torch.squeeze(input=xx_temp)
                weight=torch.sigmoid(F.linear(xx_temp,self.fc_w17,self.bias)+F.linear(x_outt,self.fc_w17,self.bias))
                xx=xx.detach()
                w1=weight.clone().expand(batch_size,growth)
                w1=w1.repeat(1,1,17)
                w1=w1.view(batch_size,growth,17,1)
                xx=w1*xx.clone()
                xxx=torch.cat((xxx,xx),1)
                i=i+1
        xxx=torch.cat((xxx,x_out),1)
        out = self.conv1(F.relu(self.bn1(xxx),inplace=True))#(x=[64, 12, 15, 3],out=[64, 6, 15, 3])
        #print('conv(x)',out.shape)
        out = torch.cat((x,out), 1)#([64, 18, 15, 3])
        del x,x1,x2,x11,xxx,weight,x_out,x_outt
        return out

class Transition(nn.Module):
  def __init__(self, nChannels, nOutChannels):
    super(Transition, self).__init__()
    self.bn1 = nn.BatchNorm2d(nChannels)
    self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=(1,1), bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x),inplace=True))#[64, 36, 15, 3]
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
    #InChannels=nChannels
    self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate
    nOutChannels = int(math.floor(nChannels*reduction))
    self.trans2 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    #InChannels=nChannels
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
        #print('调用SingleLayer')
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
    out_all = self.fc(out_all)#64,108
    out=torch.sigmoid(out_all)
    return out

def densenet100_12(num_classes=1):
  model = DenseNet(6, 36, 0.5, num_classes, False)
  return model
