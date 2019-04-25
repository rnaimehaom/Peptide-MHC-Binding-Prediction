# -*- coding: utf-8 -*-
from __future__ import division
import os, sys, pdb, shutil, time, random, copy
import argparse
import torch
import torch.backends.cudnn as cudnn
#import torchvision.datasets as dset
#import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import torch.utils.data as Data
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
#from tensorboard_logger import configure,log_value
from logger import Logger
#configure("runs/run-densenet")
logger = Logger('./logs')
model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', metavar='ARCH', default='densenet100_12', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.05, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./save', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='./checkpoint.pth.tar')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
#parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

if args.manualSeed is None:
  args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
  torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True


def main():
  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("torch  version : {}".format(torch.__version__), log)
  print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

  # Init dataset
  #if not os.path.isdir(args.data_path):
    #os.makedirs(args.data_path)

  fr = open('data/Train_data1.txt', 'rb')
  x_train = pickle.load(fr)#[？, 21, 49]
  y_train = pickle.load(fr)
  
  torch_dataset = Data.TensorDataset(x_train, y_train)
  fr = open('data/Dev_data1.txt', 'rb')
  x_val = pickle.load(fr)#[？, 21, 49]
  y_val = pickle.load(fr)
  torch_testset = Data.TensorDataset(x_val, y_val)
  
  #global y_binary
  #y_val=y_val.cpu()
  #y_binary =np.where(y_val>=0.42562, 1,0)
  num_classes = 1

  train_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(torch_testset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

  print_log("=> creating model '{}'".format(args.arch), log)
  # Init model, criterion, and optimizer
  net = models.__dict__[args.arch](num_classes)
  print_log("=> network :\n {}".format(net), log)

  net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

  # define loss function (criterion) and optimizer
  criterion = torch.nn.MSELoss()

  optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)

  if args.use_cuda:
    net.cuda()
    criterion.cuda()

  recorder = RecorderMeter(args.epochs)
  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print_log("=> loading checkpoint '{}'".format(args.resume), log)
      checkpoint = torch.load(args.resume)
      recorder = checkpoint['recorder']
      args.start_epoch = checkpoint['epoch']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
    else:
      raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
  else:
    print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

  #if args.evaluate:
  #  validate(test_loader, net, criterion, log,epoch)
  #  return

  # Main loop
  start_time = time.time()
  epoch_time = AverageMeter()
  for epoch in range(args.start_epoch, args.epochs):
    current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

    need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

    print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                + ' [Best : Auc={:.2f}]'.format(recorder.max_accuracy(False)), log)

    # train for one epoch
    train_auc, train_los = train(train_loader, net, criterion, optimizer, epoch, log)

    # evaluate on validation set
    #val_acc,   val_los   = extract_features(test_loader, net, criterion, log)
    val_auc,   val_los   = validate(test_loader, net, criterion, log,epoch)
    is_best = recorder.update(epoch, train_los, train_auc, val_los, val_auc)

    save_checkpoint({
      'epoch': epoch + 1,
      'arch': args.arch,
      'state_dict': net.state_dict(),
      'recorder': recorder,
      'optimizer' : optimizer.state_dict(),
      'args'      : copy.deepcopy(args),
    }, is_best, args.save_path, 'checkpoint.pth.tar')

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
    recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

  log.close()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  #auc=AverageMeter()
  
  # switch to train mode
  model.train()
  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if args.use_cuda:
      target = target.cuda(non_blocking=True)
      input = input.cuda()
      input_var = torch.autograd.Variable(input)
      target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    target_var=target_var.float()
    target_var=torch.squeeze(target_var,2)
    loss = criterion(output, target_var)
    
    # measure accuracy and record loss
    #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
  

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    target=target.cpu().data.numpy()
    target_binary=np.where(target>=0.42562, 1,0)
    #target_var=target_var.cpu().data.numpy()
    output=output.cpu().data.numpy()
    
   
    auc = roc_auc_score(target_binary.flatten(), output.flatten())
    
    
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
            'Time  ({batch_time.avg:.3f})   '
            'Data  ({data_time.avg:.3f})   '
            'Loss  ({loss.avg:.4f})   '
            'Auc  ({auc:.3f})   '.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses, auc=auc) + time_string(), log)
      #log_value('train_auc', auc, i)
  info = { 'train_loss': loss.item(),'train_auc': auc.item() }
  for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)
  for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(),epoch)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)
  #print_log('  **Train** Loss {loss:.3f} Auc {auc:.3f} '.format(auc=auc, loss=loss), log)
  #loss=float(losses)
  #log_value('train_loss', losses, epoch)
  
  return auc, loss

def validate(val_loader, model, criterion, log,epoch):
  losses = AverageMeter()
  
  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda(non_blocking=True)
      input = input.cuda()
    with torch.no_grad():
      input_var = torch.autograd.Variable(input)
      target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    target_var=target_var.float()
    #target_var=torch.squeeze(target_var,2)
    loss = criterion(output, target_var)
    target=target.cpu()
    target_binary=np.where(target>=0.42562, 1,0)
    output=output.cpu().data.numpy()
    auc = roc_auc_score(target_binary.flatten(), output.flatten())
    # measure accuracy and record loss
    #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    #top1.update(prec1[0], input.size(0))
    #top5.update(prec5[0], input.size(0))

  print_log('  **Test** loss@1 {loss:.3f} auc@1 {auc:.3f}'.format(loss=loss,auc=auc), log)
  #log_value('val_loss', loss, epoch)
  #log_value('val_auc', auc, epoch)
  info = {'valid_loss': loss.item(),'valid_auc': auc.item() }
  for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)
  return  auc, loss

def extract_features(val_loader, model, criterion, log):
  losses = AverageMeter()
  #top1 = AverageMeter()
  #top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  for i, (input, target) in enumerate(val_loader):
    if args.use_cuda:
      target = target.cuda(non_blocking=True)
      input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output, features = model([input_var])

    pdb.set_trace()

    loss = criterion(output, target_var)

    # measure accuracy and record loss
    #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    #top1.update(prec1.item(), input.size(0))
    #top5.update(prec5.item(), input.size(0))

  print_log('  **Test**  Loss@1 {loss:.3f}'.format( loss=loss), log)

  return losses.avg

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()

def save_checkpoint(state, is_best, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best.pth.tar')
    shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

if __name__ == '__main__':
  main()
