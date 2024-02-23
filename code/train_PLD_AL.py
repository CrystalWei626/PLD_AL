import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader,Dataset
from pylab import *
import argparse
import os

sys.path.append('data/tyc/code')
import Query_stg.query_stg as query_stg
import Dataset.Get_data as Get_data
from train.PLD_AL import PLD_AL_train

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_path', default='/data/tyc/data_CUBS/', type=str)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--max_iterations',default=1500,type=int)
parser.add_argument('--max_iterations_last',default=1000,type=int)
parser.add_argument('--Max_AL_iter',default=5,type=int)
parser.add_argument('--label_size',default=200,type=int)
parser.add_argument('--batch_size',default=14,type=int)
parser.add_argument('--labeled_bs',default=2,type=int)
parser.add_argument('--batch_size_test',default=6,type=int)
parser.add_argument('--base_lr',default=0.01,type=float)
parser.add_argument('--seed',default=1000,type=int)
parser.add_argument('--pseudo',default=True,type=bool)
parser.add_argument('--pseudo_rect',default=True,type=bool)
parser.add_argument('--ema',default=True,type=bool)
parser.add_argument('--consistency_type',default='mse',type=str)
parser.add_argument('--consistency',default=1.0,type=float)
parser.add_argument('--consistency_rampup',default=40.0,type=float)
parser.add_argument('--ema_decay',default=0.99,type=float)
parser.add_argument('--patch_size',default=[128, 256],type=list)
parser.add_argument('--T',default=1,type=float)
parser.add_argument('--lam',default=0.9,type=float)
parser.add_argument('--tau',default=0.05,type=float)
parser.add_argument('--K1',default=1,type=int)

if __name__ == '__main__':
   args = parser.parse_args()

   #beat_sup_metrics = []
   Loss = []
   #pseudo_rect = True
   #pseudo = True
   correct_save_path = args.root_path+'saved_correct/'

   torch.cuda.device_count()
   torch.cuda.is_available()
   torch.cuda.set_device(0)
   torch.backends.cudnn.enabled = True
   torch.backends.cudnn.benchmark = True

   Time = []
   train_dataset, test_dataset, unlabeled_dataset = Get_data.DataLoad(args.root_path,True)
   test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_test, num_workers=0,
                            pin_memory=False)
   for AL_time in np.arange(args.Max_AL_iter+1):
     train_loader = DataLoader(dataset=train_dataset, batch_size=args.labeled_bs, num_workers=0,
                          pin_memory=False)
     unlabeled_loader = DataLoader(dataset=unlabeled_dataset,batch_size=args.batch_size-args.labeled_bs,num_workers=0,
                                pin_memory=False)

     start_time = time.time()
     model,ema_model = PLD_AL_train(args,train_loader,unlabeled_loader,test_loader,correct_save_path,AL_time,True)
     end_time = time.time()
     Time.append(round(end_time-start_time,2))
     Unlabeled_list = unlabeled_loader.dataset.data_path
     if AL_time != (args.Max_AL_iter):
      quary_method = 'KL'
      chosen = query_stg.Query(model,ema_model,unlabeled_dataset,train_dataset,args.patch_size,args.label_size,quary_method,dropout = False)
      del model
      del ema_model
      train_dataset.update_dict(chosen, Unlabeled_list)
      unlabeled_dataset.delete_dict(chosen)
      args.labeled_bs += 2
   print(Time)