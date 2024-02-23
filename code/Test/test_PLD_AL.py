import torch
from medpy import metric
from pylab import *
import surface_distance as surfdist
import sys
sys.path.append('data/tyc/code')
import Dataset.Get_data as Get_data
#sys.path.append('data/tyc/code/Refinement')
from Refinement.Refine import compute_IOU

def validation(test_loader,model):
    metric_dice_1 = 0.0
    metric_iou_1 = 0.0
    ASD = 0.0
    HD = 0.0
    for index in np.arange(test_loader.dataset.data_path.__len__()):
        input, label_test, _ = Get_data.get_data(test_loader.dataset.data_path[index])  
        input = input.to(torch.float32)  # c, h ,w
        model.eval()
        with torch.no_grad():
            outputs = model(input.unsqueeze(0).cuda())
            outputs = outputs
            outputs_soft = torch.softmax(outputs, dim=1)
            out = torch.argmax(outputs_soft, dim=1).squeeze(0).cpu()
            metric_dice_1 += np.sum(metric.binary.dc(out.numpy() == 1, np.array(label_test) == 1))
            metric_o, metric_o_1 = compute_IOU(out.numpy(), np.array(label_test))
            metric_iou_1 += metric_o_1

            surface_distances = surfdist.compute_surface_distances(
                label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                spacing_mm=(1.0, 1.0))
            avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
            ASD += avg_surf_dist[1]
            surface_distances = surfdist.compute_surface_distances(
                label_test.detach().numpy().astype('bool'), out.detach().numpy().astype('bool'),
                spacing_mm=(1.0, 1.0))
            hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)  # 95HD
            HD += hd_dist_95
    metric_dice_1 = metric_dice_1 / test_loader.dataset.data_path.__len__()
    metric_iou_1 = metric_iou_1 / test_loader.dataset.data_path.__len__()
    ASD = ASD / test_loader.dataset.data_path.__len__()
    HD = HD / test_loader.dataset.data_path.__len__()

    return  metric_dice_1,metric_iou_1,ASD,HD