import util
import os
import argparse
from model import *
import numpy as np
import pandas as pd
from util import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=50,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--checkpoint',type=str,help='',default='./garage/metr_exp1_best_3.32.pth')
parser.add_argument('--plotheatmap',type=str,default='False',help='')
args = parser.parse_args()
def pred(input_x):
    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    model = gwnet(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool,
                  addaptadj=args.addaptadj, aptinit=adjinit)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print('model load successfully')
    # 计算均值和方差，训练测试保持一致，这里选择的是训练集所有的均值和方差
    # 训练的时候，数据减均值除方差，预测输出的结果乘方差+均值复原
    data = {}
    cat_data = np.load(os.path.join('data/METR-LA', 'train' + '.npz'))
    data['x_' + 'train'] = cat_data['x']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # input_x = torch.randn(1, 24, 50, 2).to(device)
    input_x = torch.Tensor(input_x).unsqueeze(dim=0)
    #数据预处理---归一化
    input_x[...,0] = scaler.transform(input_x[...,0])
    # print(input_x)
    # 预测的时候仅输出下一个预测时间段，结果进行维度调整transpose(1,3)，将预测的长度
    # 比如24放在最后，便于输出分时的结果
    input_x = input_x.transpose(1, 3).to(device)
    # print('--------', input_x.shape)

    with torch.no_grad():
        # [b_size,50,24]
        pred = model(input_x).transpose(1, 3).squeeze()

    # 结果恢复真实格式
    # view需要地址连续，要加上.contiguous()
    pred = scaler.inverse_transform(pred).contiguous()
    if 0:
        save_data = pred.cpu().detach().view(1,-1).squeeze().numpy()
        df2 = pd.DataFrame({'pred': save_data})
        df2.to_csv('./my_pred.csv',index=False)
    else:
        save_data = pred.transpose(0,1).cpu().detach().numpy()
        df2 = pd.DataFrame(save_data)
        df2.to_csv('./my_pred.csv')

def main():
    test = np.load(r'D:\code_yjw\traffic_predection\Graph-WaveNet-master\data_\METR-LA' + '/' + 'test.npz')
    #取测试的第一个数据进行测试，数据维度是[12,207,2]
    input_x = test['x'][0]
    pred(input_x)

if __name__ == "__main__":
    main()
