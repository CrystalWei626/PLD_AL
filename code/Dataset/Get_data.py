import os.path
import torch
from torch.utils.data import DataLoader,Dataset
import h5py
from PIL import Image
from pylab import *
import cv2

def get_data(data_path):
    h5f = h5py.File(data_path)
    image = Image.fromarray(np.array(h5f['image'][()]).astype('uint8'))
    image = np.array(image).reshape(1, image.size[1], image.size[0]).astype(np.float32)
    label = Image.fromarray(np.array(h5f['label'][()]).astype('uint8'))
    seg_label = Image.fromarray(np.array(h5f['label_rect'][()]).astype('uint8'))
    h5f.close()
    return torch.from_numpy(np.array(image)), torch.from_numpy(np.array(label)).long(), \
        torch.from_numpy(np.array(seg_label)).long()


class FullDataset(Dataset):
    def __init__(self, root_path, Init, type='Train', transform=None):
        self.transform = transform
        self.type = type
        if self.type == 'Train':
            self.label_dataset_dir = os.path.join(root_path, 'train')
        if self.type == 'Test':
            self.label_dataset_dir = os.path.join(root_path, 'test')
        if self.type == 'Unlabel':
            self.label_dataset_dir = os.path.join(root_path, 'unlabeled')
        if self.type == 'Full_Train':
            self.label_dataset_dir = os.path.join(root_path, 'full_train')
        self.dir_list = os.listdir(self.label_dataset_dir)
        self.transform = transform
        self.data_path = []
        self.Correct_Time = []
        if Init == True:
            self.init_seg_dict(self.type)

    def __getitem__(self, index):
        h5f = h5py.File(self.data_path[index])
        image = Image.fromarray(np.array(h5f['image'][()]).astype('uint8'))
        image = np.array(image).reshape(1, image.size[1], image.size[0]).astype(np.float32)
        label = Image.fromarray(np.array(h5f['label'][()]).astype('uint8'))
        label_rect = Image.fromarray(np.array(h5f['label_rect'][()]).astype('uint8'))
        sample = {'image': torch.from_numpy(np.array(image)), 'label': torch.from_numpy(np.array(label)).long(),
                  'label_rect': torch.from_numpy(np.array(label_rect)).long()}
        h5f.close()
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.data_path.__len__()

    def init_seg_dict(self, type):
        for idx in range(len(self.dir_list)):
            self.Correct_Time.append(0)
            self.data_path.append(os.path.join(self.label_dataset_dir, self.dir_list[idx]))

    def Cauculate_KL(self,T,pseudo_rect,threshold_pseudo):
        pred1 = self.prev_pred_prob_dict
        pred2 = self.prev_pred_prob_dict_ema
        length = pred1.__len__()
        pred_1 = pred1[0]
        pred_2 = pred2[0]
        for i in np.arange(1, length):
            pred_1 = torch.cat((pred_1, pred1[i]), dim=0)
            pred_2 = torch.cat((pred_2, pred1[i]), dim=0)
        pseudo_label = torch.softmax(pred_2.detach() / T, dim=1)
        max_probs, targets = torch.max(pseudo_label, dim=1)
        if pseudo_rect:
            self.pseudo_label_rect, self.KL_Dis = update_variance(pred_1, pred_2, targets)
        else:
            mask = max_probs.ge(threshold_pseudo).float()
            self.pseudo_label_rect = targets * mask

    def update_dict(self, chosen, data_path):
        for c in chosen:
            self.data_path.append(data_path[c])
            self.Correct_Time.append(0)

    def delete_dict(self, chosen):
        chosen.sort()
        for index in chosen[::-1]:
            del self.data_path[index]

    # the core logic to update the pseudo annotation
    def update_seg_dict(self, idx, IoU_npl_indx, outputs_soft, out, correct_save_path,mask_threshold=0.8):
        if self.Correct_Time[idx] <= 100:
            self.update_allclass(idx, IoU_npl_indx, outputs_soft, out, mask_threshold, 'single',correct_save_path, class_constraint=True,
                                 update_or_mask='update', update_all_bg_img=True)
            self.Correct_Time[idx] += 1

    def update_allclass(self, idx, IoU_npl_indx, outputs_soft, out, mask_threshold, IoU_npl_constraint,correct_save_path,
                        class_constraint=True,
                        update_or_mask='update', update_all_bg_img=True):
        Img, label, seg_label = get_data(self.data_path[idx])  # h,w
        h, w = seg_label.size()  # h,w
        b = 1

        seg_argmax, seg_prediction_prob = out, outputs_soft
        seg_prediction_max_prob = seg_prediction_prob.max(axis=1)[0]

        if class_constraint == True and (set(np.unique(seg_label[0].numpy())) - set(np.array([0, 255]))):
            for i_batch in range(b):
                unique_class = torch.unique(seg_label[i_batch])
                # print(unique_class)
                indx = torch.zeros((h, w), dtype=torch.long)
                for element in unique_class:
                    indx = indx | (seg_argmax[i_batch] == element)
                seg_argmax[i_batch][(indx == 0)] = 1

        seg_label = seg_label.cpu()
        seg_argmax = seg_argmax.cpu()
        seg_prediction_max_prob = seg_prediction_max_prob.cpu()
        seg_change_indx = (seg_label != seg_argmax) & (
                seg_prediction_max_prob > mask_threshold)

        if IoU_npl_constraint == 'both':
            class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)
            class_indx_seg_label = torch.zeros((b, h, w), dtype=torch.bool)

            for element in IoU_npl_indx:
                class_indx_seg_argmax = class_indx_seg_argmax | (seg_argmax == element)
                class_indx_seg_label = class_indx_seg_label | (seg_label == element)
            seg_change_indx = seg_change_indx & class_indx_seg_label & class_indx_seg_argmax

        elif IoU_npl_constraint == 'single':
            class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)

            for element in IoU_npl_indx:
                class_indx_seg_argmax = class_indx_seg_argmax | (seg_argmax == element)
            seg_change_indx = seg_change_indx & class_indx_seg_argmax

        seg_label_clone = seg_label.clone().unsqueeze(0).unsqueeze(0)
        seg_label_clone[seg_change_indx] = seg_argmax[seg_change_indx]
        if torch.sum(seg_label_clone != 0) < 0.5 * torch.sum(seg_label != 0) and torch.sum(seg_label_clone == 0) / (
                b * h * w) > 0.95:
            return

        if update_or_mask == 'update':
            seg_label[seg_change_indx.squeeze(0).squeeze(0)] = seg_argmax[seg_change_indx]
        else:
            seg_label[seg_change_indx] = (torch.ones((b, h, w), dtype=torch.long) * 255)[
                seg_change_indx]

        h5f = h5py.File(self.data_path[idx], "r+")
        h5f.__delitem__("label_rect")
        h5f['label_rect'] = seg_label.cpu().detach().numpy()
        h5f.close()
        pre = seg_label.cpu().detach().numpy() * 255
        cv2.imwrite(correct_save_path + str(idx) + '_correct_{}.png'.format(self.Correct_Time[idx]), pre.astype('int'))


def DataLoad(root_path, Init):
    train_dataset = FullDataset(root_path, Init, type='Train')
    test_dataset = FullDataset(root_path, Init, type='Test')
    unlabeled_dataset = FullDataset(root_path, Init, type='Unlabel')
    return train_dataset, test_dataset, unlabeled_dataset