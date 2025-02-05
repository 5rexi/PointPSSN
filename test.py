import numpy as np
import open3d as o3d
import torch

from models.PointNet import PointNet
from models.PointNet2 import PointNet2
from models.DGCNN import DGCNN
from models.RepSurf import RepSurf
from models.PointMLP import PointMLP
from models.PointNeXT import PointNeXT
from models.PointPSSN import PointPSSN

model_PointNet = PointNet(input_channel=3, output_channel=2)
model_PointNet.load_state_dict(torch.load('./weights/PointNet.pth'))
model_PointNet.eval()
model_PointNet2 = PointNet2(num_classes=2)
model_PointNet2.load_state_dict(torch.load('./weights/PointNet++.pth'))
model_PointNet2.eval()
model_DGCNN = DGCNN(input_channel=3, output_channel=2)
model_DGCNN.load_state_dict(torch.load('./weights/DGCNN.pth'))
model_DGCNN.eval()
model_RepSurf = RepSurf(input_channel=3, output_channel=2)
model_RepSurf.load_state_dict(torch.load('./weights/RepSurf.pth'))
model_RepSurf.eval()
model_PointMLP = PointMLP(2)
model_PointMLP.load_state_dict(torch.load('./weights/PointMLP.pth'))
model_PointMLP.eval()
model_PointPSSN = PointPSSN(num_classes=2)
model_PointPSSN.load_state_dict(torch.load('./weights/PointPSSN_acc_best.pth'))
model_PointPSSN.eval()
model_PointNeXT = PointNeXT(num_classes=2)
model_PointNeXT.load_state_dict(torch.load('./weights/PointNeXT.pth'))
model_PointNeXT.eval()


def load_file(path):
    data = np.loadtxt(path)
    sampled_points = data[np.random.choice(data.shape[0], 2048, replace=False)]
    return sampled_points

def test(data, model_name, show=True):
    data_copy = data
    data = torch.from_numpy(data).float()
    data = data.unsqueeze(0)
    data = data.transpose(1, 2)
    result = None
    if model_name == 'PointNet':
        model = model_PointNet
        result = model(data)
    elif model_name == 'PointNet++':
        model = model_PointNet2
        result = model(data)
    elif model_name == 'DGCNN':
        model = model_DGCNN
        result = model(data)
    elif model_name == 'RepSurf':
        model = model_RepSurf
        result = model(data)
    elif model_name == 'PointMLP':
        model = model_PointMLP
        result = model(data)
    elif model_name == 'PointPSSN':
        model = model_PointPSSN
        result = model(data)
    elif model_name == 'PointNeXT':
        model = model_PointNeXT
        result = model(data)
    else :
        print('Error')

    if show:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data_copy)
        color = np.zeros([2048, 3])
        result = result.transpose(1, 2)
        for i in range(2048):
            if result[0][i][0] > result[0][i][1]:
                color[i] = [169 / 255, 181 / 255, 209 / 255]
            else :
                color[i] = [223 / 255, 166 / 255, 95 / 255]
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd])

# for i in range(47, 100):
#     print(i)
#     path = './dataset/Data/test/data_' + str(i + 1) + '.txt'
#     data = load_file(path)
#     model_list = ['PointPSSN', 'PointMLP', 'PointNet', 'PointNet++', 'DGCNN', 'RepSurf', 'PointNeXT']
#     for j in range(7):
#         print(model_list[j])
#         test(data, model_list[j])
#         if j == 1:
#             a = input('OK? (Y/n)')
#             if a == 'N' or a == 'n':
#                 break
#             elif a == 'Y' or a == 'y':
#                 for k in range(7):
#                     print(model_list[k])
#                     test(data, model_list[k])
#                 break
#             else:
#                 exit(10)
from thop import profile
input = torch.randn(6, 3, 2048)
# flops, params = profile(model_PointNet, inputs=(input, ))
# print('PointNet', flops, params)
# flops, params = profile(model_PointNet2, inputs=(input, ))
# print('PointNet2', flops, params)
# flops, params = profile(model_DGCNN, inputs=(input, ))
# print('DGCNN', flops, params)
# flops, params = profile(model_RepSurf, inputs=(input, ))
# print('RepSurf', flops, params)
# flops, params = profile(model_PointMLP, inputs=(input, ))
# print('PointMLP', flops, params)
flops, params = profile(model_PointPSSN, inputs=(input, ))
print('PointPSSN', flops, params)
flops, params = profile(model_PointNeXT, inputs=(input, ))
print('PointNeXT', flops, params)


