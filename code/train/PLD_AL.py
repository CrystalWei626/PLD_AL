import os.path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
import h5py
import segmentation_models_pytorch as smp
from pylab import *
import cv2

sys.path.append('data/tyc/code')
import train.losses as losses
from train.update_loss import get_current_consistency_weight,update_consistency_loss,update_ema_variables
#sys.path.append('data/tyc/code/Dataset')
import Dataset.Get_data as Get_data
#sys.path.append('data/tyc/code/Refinement')
from Refinement.Refine import compute_IOU,compute_iou,update_iou_stat,if_update
#sys.path.append('data/tyc/code/test')
from Test.test_PLD_AL import validation

torch.cuda.device_count()
torch.cuda.is_available()
torch.cuda.set_device(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



def write_in_h5(h5_save_path,images,labels,label_rect):
    with h5py.File(h5_save_path, mode='w') as hf:
        hf.create_dataset(name="image", dtype=float, data=images)
        hf.create_dataset(name="label", dtype=float, data=np.array(labels).astype(bool).astype(int))
        hf.create_dataset(name="label_rect", dtype=float, data=np.array(label_rect).astype(bool).astype(int))
    print(h5_save_path)


def PLD_AL_train(args,train_loader,unlabeled_loader,test_loader,correct_save_path,AL_time,Rect):
    update_sign_list = []
    Updated_class_list = []
    Abs_change_1 = []
    Abs_change_0 = []

    IoU_npl_dict = {}
    for i in range(2):
        IoU_npl_dict[i] = []
    updated = False
    metric_list = 0.0

    model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_depth=4, encoder_weights='imagenet',
                                 decoder_use_batchnorm=True,
                                 decoder_channels=[128, 64, 32, 16], in_channels=1, classes=args.num_classes,
                                 activation='tanh', aux_params=None)
    ema_model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_depth=4,encoder_weights='imagenet',
                                 decoder_use_batchnorm=True,
                                 decoder_channels=[128, 64, 32, 16], in_channels=1, classes=args.num_classes,
                                 activation='tanh', aux_params=None)
    for param in ema_model.parameters():
          param.detach_()
    ema_model = ema_model.cuda()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    best_performance = 0.0
    if AL_time == (args.Max_AL_iter):
        max_iter = args.max_iterations_last
    else:
        max_iter = args.max_iterations
    max_epoch = max_iter // len(train_loader)
    iterator = tqdm(range(max_epoch), ncols=70)
    writer = SummaryWriter(args.root_path + '/log')
    logging.basicConfig(filename=args.root_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(len(train_loader)))
    for epoch_num in iterator:
        TP_npl_epoch = [0] * 2
        P_npl_epoch = [0] * 2
        T_npl_epoch = [0] * 2
        for i_batch, sampled_batch in enumerate(zip(train_loader,unlabeled_loader)):
            img_batch = torch.cat((sampled_batch[0]['image'].to(torch.float32),sampled_batch[1]['image'].to(torch.float32)),
                                    dim=0)
            label_rect = sampled_batch[0]['label_rect']
            lab = sampled_batch[0]['label']
            if torch.cuda.is_available():
              img_batch = img_batch.cuda()
              label_batch = label_rect.cuda().long()

            noise = torch.clamp(torch.randn_like(img_batch) * 0.1, -0.2, 0.2)

            student_inputs = img_batch + noise
            ema_inputs = img_batch

            outputs = model(student_inputs)

            outputs_soft = torch.softmax(outputs, dim=1)
            out = torch.argmax(outputs_soft, dim=1).unsqueeze(1)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs.cuda())

            label_np_updated = label_batch.cpu().numpy()
            label_b = label_batch.shape[0]
            TP_npl_epoch, P_npl_epoch, T_npl_epoch = update_iou_stat(out[:label_b].squeeze(1).detach().cpu().numpy(),
                                                                     label_np_updated, TP_npl_epoch,
                                                                     P_npl_epoch, T_npl_epoch)

            loss_seg = F.cross_entropy(outputs[:label_b], label_batch[:label_b])
            loss_seg_dice = losses.dice_loss(outputs_soft[:label_b, 1, :, :], label_batch[:label_b] == 1) + \
                            losses.dice_loss(outputs_soft[:label_b, 0, :, :], label_batch[:label_b] == 0)
            supervised_loss = (loss_seg + 1.25*loss_seg_dice)


            consistency_weight = get_current_consistency_weight(iter_num // 150,args.consistency,args.consistency_rampup)

            consistency_loss = update_consistency_loss(outputs, ema_output,args.pseudo,args.pseudo_rect,args.T)
            consistency_dist = consistency_weight * consistency_loss
            loss = supervised_loss + consistency_dist * 100


            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2) #这是控制梯度的大小二范数不超�?0
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = args.base_lr * (1.0 - iter_num / (max_iter)) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num +1
            logging.info('iteration %d : loss : %f cons_dist: %f, loss_entropy: %f,loss_dice: %f' %
                         (iter_num, loss.item(), consistency_dist.item(),
                          loss_seg.item(),
                          loss_seg_dice.item()))

            print('-' * 50)

            # Validation
            if iter_num > args.max_iterations/2 and iter_num % 100 == 0: #and AL_time == (Max_AL_iter):
                metric_dice_1,metric_iou_1,ASD,HD = validation(test_loader,ema_model)

                writer.add_scalar('info/val_dice',
                                  metric_list, iter_num)
                performance = metric_dice_1
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(args.root_path,
                                                  'AL_iter_{}_dice_{}_{}_{}_{}.pth'.format(
                                                      iter_num, round(best_performance, 4), round(metric_iou_1, 4),
                                                      round(ASD, 4),
                                                      round(HD, 4)))
                    save_best = os.path.join(args.root_path,
                                             'AL_{}_model_{}.pth'.format(AL_time, args.label_size))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                logging.info(
                    'iteration %d : mean_dice : %f' % (iter_num, performance))

                #model.train()
        if AL_time != (args.Max_AL_iter) and Rect == True:
         IoU_npl_epoch = compute_iou(TP_npl_epoch,P_npl_epoch,T_npl_epoch)
         for i in range(2):
            IoU_npl_dict[i].append(IoU_npl_epoch[i])
         if updated==False and epoch_num>=args.K1:
           IoU_npl_indx = [0]
           update_sign_1,abs_change_1 = if_update(np.array(IoU_npl_dict[1]), epoch_num, len(train_loader),threshold=args.tau)  # 前景标签的更�?
           Abs_change_1.append(abs_change_1)
           update_sign_0,abs_change_0 = if_update(np.array(IoU_npl_dict[0]), epoch_num, len(train_loader), threshold=args.tau)  # 背景标签的更�?
           Abs_change_0.append(abs_change_0)
           update_sign_list.append([update_sign_0,update_sign_1])
           if update_sign_0 == True or update_sign_1==True:
              for index in np.arange(train_loader.dataset.data_path.__len__()):
                   input,label,label_n = Get_data.get_data(train_loader.dataset.data_path[index])
                   input = input.to(torch.float32) #c, h ,w
                   model.eval()
                   ema_model.eval()
                   with torch.no_grad():
                       outputs = model(input.unsqueeze(0).cuda())
                       ema_outputs = ema_model(input.unsqueeze(0).cuda())
                       outputs_soft = torch.softmax(ema_outputs, dim=1)
                       out = torch.argmax(outputs_soft, dim=1).unsqueeze(1).cpu()
                   if AL_time == 0:
                       img = input.squeeze(0).squeeze(0).detach().cpu().numpy()
                       #cv2.imwrite(correct_save_path + str(index) + '_image.png', img.astype('int'))
                       label_gt = label.squeeze(0).squeeze(0).detach().cpu().numpy()
                       lab = label_gt * 255
                       #cv2.imwrite(correct_save_path + str(index) + '_label.png', lab.astype('int'))
                       label_noise = label_n.squeeze(0).squeeze(0).detach().cpu().numpy()
                       lab_noise = label_noise * 255
                       #cv2.imwrite(correct_save_path + str(index) + '_label_noise.png', lab_noise.astype('int'))
                   if update_sign_0 == True and 0 not in Updated_class_list:
                     IoU_npl_indx = [0]
                     train_loader.dataset.update_seg_dict(index,IoU_npl_indx,outputs_soft,out,correct_save_path, mask_threshold=args.lam)
                   if update_sign_1 == True and 1 not in Updated_class_list:
                     IoU_npl_indx = [1]
                     train_loader.dataset.update_seg_dict(index,IoU_npl_indx,outputs_soft,out,correct_save_path,mask_threshold=args.lam)
              if update_sign_0 == True and 0 not in Updated_class_list:
                  Updated_class_list.append(0)
              if update_sign_1 == True and 1 not in Updated_class_list:
                  Updated_class_list.append(1)
           if 0 in Updated_class_list and 1 in Updated_class_list:
             updated = True
             metric_dice_1, metric_iou_1, ASD, HD = validation(test_loader, ema_model)
             save_Time = os.path.join(args.root_path,'AL_{}_dice_{}_{}_{}_{}.pth'.format(AL_time,metric_dice_1,metric_iou_1,ASD,HD))
             torch.save(model.state_dict(), save_Time)
             #print(Abs_change_0)
             #print(Abs_change_1)
             #f = open(os.path.join(args.root_path,"Abs_change_0_{}.txt".format(AL_time)), "w")
             #f.writelines(str(Abs_change_0))
             #f.close()
             #f = open(os.path.join(args.root_path,"Abs_change_1_{}.txt".format(AL_time)), "w")
             #f.writelines(str(Abs_change_1))
             #f.close()
             break
    #beat_sup_metrics.append(best_performance)
    return model,ema_model

