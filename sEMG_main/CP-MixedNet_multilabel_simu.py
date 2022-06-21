# from tkinter import _Padding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import scikitplot as skplt
from matplotlib import pyplot as plt

from tensorboardX import SummaryWriter
import numpy as np
import torch.utils.data.dataloader as DataLoader
from datasets import DataLoader as DL
from argparse import ArgumentParser

highest_F1 = [ 0 for i in range(7)]

class EMG_NET_1(nn.Module):  # Firstly attention_project different channel
    def __init__(self, channel_p=21, channel_temp=25, conv_pers_window=11, pool_pers_window=3,conv_degree_window=7
                 ,pool_pers_degree=3, window=100, target_dim=8):
        super(EMG_NET_1, self).__init__()

        self.target_dim = target_dim
        self.channel_p = channel_p
        self.channel_temp = channel_temp
        self.conv_pers_window = conv_pers_window
        self.pool_pers_window = pool_pers_window
        self.conv_pers_degree = conv_degree_window
        self.pool_pers_degree = pool_pers_degree
        self.window = window
        # The input size is (trails * channels * 1 * time_indexs)
        # ***CP-Spatio-Temporal Block***
        # Channel Projection
        self.channelProj = nn.Conv2d(4, self.channel_p, 1, stride=1, bias=False)  # (7*1*index)->(21*1*index)
        self.batchnorm_proj_tranf = nn.BatchNorm2d(self.channel_p)
        self.batchnorm_proj_tranf_2 = nn.BatchNorm2d(self.channel_p)
        # Shape Transformation
        self.shapeTrans = nn.Conv2d(self.channel_p, self.channel_p, 1, stride=1,
                                    bias=False)  # (21*1*2700)->(21*1*2700)  这个卷积有什么必要？
        # Temporal Convolution

        self.conv1 = nn.Conv2d(1, self.channel_temp, (1, self.conv_pers_window), stride=1,
                               bias=False)  # (1*21*2700)->(25*21*2690)   #TIME-feature extraction
        self.batchnorm1 = nn.BatchNorm2d(self.channel_temp, False)
        # Spatial Convolution
        self.conv2 = nn.Conv2d(self.channel_temp, self.channel_temp, (self.channel_p, 1), stride=1, padding=0,
                               bias=False)  # (25*21*2690)->(25*2*2690)   #spatial-feature extraction
        self.batchnorm2 = nn.BatchNorm2d(self.channel_temp, False)
        # Max Pooling
        self.maxPool1 = nn.MaxPool2d((1, self.pool_pers_window), stride=(1, self.pool_pers_window),
                                     padding=0)

        # ***MS-Conv Block***
        # unDilated Convolution
        self.conv3 = nn.Conv2d(self.channel_temp, self.channel_temp, (1, self.conv_pers_window), stride=1,
                               padding=(0, (self.conv_pers_window - 1) // 2), bias=False)  # (25*2*893)->(25*2*893)
        self.batchnorm3 = nn.BatchNorm2d(self.channel_temp)
        # Dilated Convolution
        self.dilatedconv = nn.Conv2d(self.channel_temp, self.channel_temp, (1, self.conv_pers_window), stride=1,
                                     padding=(0, self.conv_pers_window - 1), dilation=2,
                                     bias=False)  # (25*2*893)->(25*2*893)
        self.batchnormDil = nn.BatchNorm2d(self.channel_temp)
        # Max pooling after Concatenating
        self.batchnorm_cancat = nn.BatchNorm2d(1, 3 * self.channel_temp - self.conv_pers_degree + 1)
        # convolution after concatenated across degree and time-scale
        self.conv_cat_degree = nn.Conv2d(1, 1, (self.conv_pers_degree, self.conv_pers_window), bias=False)
        self.poolConcatenated = nn.MaxPool2d((self.pool_pers_degree, 1), stride=(self.pool_pers_degree//2, 1),
                                             padding=0)  # (75*1*893)->(75*1*267)

        # ***Classification Block***
        # self.conv5 = nn.Conv2d(1, 1, (self.conv_pers_degree, 1), bias=False,
                            #   stride=1)  # (75*2*267)->(75*2*257)
        # self.batchnorm5 = nn.BatchNorm2d(1, (3 * self.channel_temp - self.conv_pers_degree + 1) // self.pool_pers_degree + 1 - self.conv_pers_degree)
        # self.maxPool2 = nn.MaxPool2d((self.pool_pers_degree, 1), stride=(self.pool_pers_degree//2, 1),
                               #      padding=0)  # (75*2*257)->(75*2*86)

        self.fc_dim = (((3 * self.channel_temp - (self.conv_pers_degree - 1)) - self.pool_pers_degree) //
                         (self.pool_pers_degree//2) + 1 )  * \
                      (((self.window - (self.conv_pers_window-1))//self.pool_pers_window)-(self.conv_pers_window-1))


        # 方案一：直接将原本2*n的设计取消，改为1*n
        self.fc = nn.Linear(self.fc_dim, self.target_dim,
                            bias=False)  # (1*6450)->(1*7)  注意此处的7指的是自由度的7，而最初始channel的7是贴片的7


        # weight initialization （可以进行记录：conv的 kernel全部初始化为正态分布，batchnorm：weight = 1，bias = 0）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #print("input size:",x.shape)
        x = F.elu(self.batchnorm_proj_tranf(self.channelProj(x)))
        #print('Channel Projection:',x.shape)
        x = F.elu(self.batchnorm_proj_tranf_2(self.shapeTrans(x)))
        # print('before Shape Transformation:',x)
        x = torch.transpose(x, 1, 2)  # 交换轴
        #print('after Shape Transformation:',x.shape)
        # x = self.drop1(x)
        # print('Temporal convolution 1:', x)
        x = self.conv1(x)
        #print('Temporal convolution 2:', x.shape)
        x = self.batchnorm1(x)
        #print('Temporal convolution 3:', x.shape)
        x = F.elu(x)
        # x = F.elu(self.batchnorm1(self.conv1(self.drop1(x))))
        # print('Temporal convolution 4:',x)
        x = F.elu(self.batchnorm2(self.conv2(x)))
        #print('Spatial convolution:',x.shape)
        x = self.maxPool1(x)
        #print('Max pooling：',x.shape)

        # x1 = F.elu(self.batchnorm3(self.conv3(self.drop3(x))))
        x_dilated = F.elu(self.batchnormDil(self.dilatedconv(x)))
        #print('Dilated Convolution1:', x_dilated.shape)
        x_undilated = F.elu(self.batchnorm3(self.conv3(x)))
        #print('Undilated Convolution2:', x_undilated.shape)

        x = torch.cat((x, x_dilated, x_undilated), dim=1)
        #print('Concatenated:', x.shape)
        x = torch.transpose(x, 1, 2)  # 交换轴
        #print('Transpose after Concatenated:',x.shape)
        x = self.conv_cat_degree(x)
        #print('Conv across degree and time',x.shape)


        x = F.elu(self.poolConcatenated(self.batchnorm_cancat(x)))
        # print('MixedScaleConv:', x.shape)

        #x = F.elu(self.batchnorm5(self.conv5(x)))
        #print('Conv5:', x.shape)
        # x = self.maxPool2(x)
        #print('maxPool2:', x.shape)
        x = x.view(-1,  self.fc_dim)
        #print('beforeFC:', x.shape)
        # print(self.fc_dim)
        x = self.fc(x)
        #print('FC:', x.shape)
        # x = self.drop5(x)          # 若数据集可以更丰富，加入更多训练者的运动数据，这里可以使用dropout
        # x = F.log_softmax(x, dim=1)      # 如果使用方案一，全部变为一位向量，loss使用MSE,则此处不需要使用softmax，softmax只用于同一自由度的两个channel之间
        # print("softmax:",x.shape)
        return x
        # 模型结尾使用了softmax函数，因此损失函数使用NLLloss()，softmax应该作用于2的维度


def binary_confusion_matrix(acts, pres, target_dim):  # 混淆矩阵代码
    TP = [0 for i in range(0, target_dim)]
    FP = [0 for i in range(0, target_dim)]
    FN = [0 for i in range(0, target_dim)]
    TN = [0 for i in range(0, target_dim)]
    for i in range(len(acts)):
        for j in range(len(acts[i])):

            if acts[i][j].item() == 1 and pres[i][j].item() == 1:
                TP[j] += 1

            elif acts[i][j].item() == 0 and pres[i][j].item() == 1:
                FP[j] += 1

            elif acts[i][j].item() == 1 and pres[i][j].item() == 0:
                FN[j] += 1

            elif acts[i][j].item() == 0 and pres[i][j].item() == 0:
                TN[j] += 1

    return TP, FP, TN, FN

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return print({'Total': total_num, 'Trainable': trainable_num})

def train(model, device, train_loader, optimizer, epoch, loss_fn, threshold, target_dim,
          log_interval=100, ):  # 每过100个batch输出一次观察，这样至少需要12800个数据，但并不存在如此多数据，因此一般只有在batch_idx=0时才会输出一次观察
    model.train()
    correct = 0
    correct_1 = 0
    all_abs = 0
    # loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
    # loss_fn = torch.nn.NLLLoss()
    loss_fn = loss_fn

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        output = output.to(torch.float32)
        target = target.to(torch.float32)  # 注意在BCEloss中output和target的类型必须相同
        # print(output)
        # print(target.shape)
        loss = loss_fn(output, target)
        # print(loss)
        # loss = F.nll_loss(output, target.squeeze())  #target 的 shape多了1维

        # loss_fun = nn.CrossEntropyLoss()
        # loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()
        '''
        for i,(name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name, param, 0)
                writer.add_scalar('loss', loss, i)
                loss = loss * 0.5
        '''
        pred = np.where(output.cpu() > threshold, 1, 0)

        pred = torch.from_numpy(pred).to(device)
        correct += pred.eq(target.view_as(pred)).sum().item()
        # print(pred)
        # print(target)
        correct_1 += (pred * target.view_as(pred)).sum().item()
        all_abs += target.sum().item()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss:{:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))

            # print(pred)
            # print(target)
            '''
            writer.add_scalar(
                "Training loss",
                loss.item(),
                epoch * len(train_loader)
            )
            '''
    print("Trainning accuracy:", 100. * correct / (len(train_loader.dataset) * target_dim))
    print("Trainning abs accuracy:", 100. * correct_1 / all_abs)


def val(model, device, val_loader, optimizer, threshold):
    model.train()
    val_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # loss = F.nll_loss(output, target.squeeze())

        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()

        val_loss += loss * batch_size
        pred = np.where(output.cpu() > threshold, 1, 0)
        # print(pred)
        correct = output.eq(target.view_as(pred)).sum().item()
        pred = torch.from_numpy(pred).to(device)
        correct += pred.eq(target.view_as(
            pred)).sum().item()  # view_as reshape the [target] and eq().sum().item()  get the sum of the correct validation
        if batch_idx == 0:
            print("pred:", output[0])
            print("true:", target[0])
    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def test(model, device, test_loader, threshold,
         target_dim, Prec_top=None, Recall_top=None, F1_top=None,
         Accu_top=None):  # 每过100个batch输出一次观察，这样至少需要12800个数据，但并不存在如此多数据，因此一般只有在batch_idx=0时才会输出一次观察
    model.eval()

    get_parameter_number(model)

    correct = 0
    correct_1 = 0
    all_abs = 0
    TP_all = [0 for i in range(0, target_dim)]
    FP_all = [0 for i in range(0, target_dim)]
    FN_all = [0 for i in range(0, target_dim)]
    TN_all = [0 for i in range(0, target_dim)]

    target_all = []
    pred_all = []


    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            output = output.to(torch.float32)
            # target = target.to(torch.float32)

            '''
            for i,(name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name, param, 0)
                    writer.add_scalar('loss', loss, i)
                    loss = loss * 0.5
            '''
            pred = np.where(output.cpu() > threshold, 1, 0)
            # pred = output.argmax(dim=1, keepdim=False)
            # print(pred)
            # correct += pred.eq(target.argmax(dim=1, keepdim=False).view_as(pred)).sum().item()
            pred = torch.from_numpy(pred).to(device)
            correct += pred.eq(target.view_as(
                pred)).sum().item()  # view_as reshape the [target] and eq().sum().item()  get the sum of the correct validation
            correct_1 += (pred * target.view_as(pred)).sum().item()
            all_abs += target.sum().item()

            # print(pred.shape)
            # print(target.shape)

            TP, FP, TN, FN = binary_confusion_matrix(target, pred, target_dim)
            for i in range(target_dim):
                TP_all[i] += TP[i]
                FP_all[i] += FP[i]
                TN_all[i] += TN[i]
                FN_all[i] += FN[i]

            # print(pred.shape)
            if batch_idx == 0:
                pred_all = pred
                target_all = target

            else:
                pred_all = torch.cat((pred_all, pred), dim=0)
                target_all = torch.cat((target_all, target), dim=0)


    Accu_abs = 100. * correct_1 / all_abs
    Accu = 100. * correct / (len(test_loader.dataset) * target_dim)
    for i in range(target_dim):
        print('DEGREE:{}--------------------------- '.format(i+1))
        print("Confusion Matrix:TP:{}  FP:{}  TN:{}  FN:{}".format(TP_all[i], FP_all[i], TN_all[i], FN_all[i]))
        if (TP_all[i] + FP_all[i]): print('Precision:', (TP_all[i]) / (TP_all[i] + FP_all[i]))
        if (TP_all[i] + FN_all[i]): print("TPR = Recall: ", TP_all[i] / (TP_all[i] + FN_all[i]))
        if (2 * TP_all[i] + FP_all[i] + FN_all[i]):
            print('F1:', (2 * TP_all[i]) / (2 * TP_all[i] + FP_all[i] + FN_all[i]))

            torch.save(model.state_dict(), "./saved_models/multi_simu/degree_v2/weights_{}.pt".format(i+1))       # 针对某一个自由度的最高F1模型
            torch.save(model, "./saved_models/whole_models/multi_degree{}_v2.pt".format(i+1))

            # plot the confusion matrix of model
            skplt.metrics.plot_confusion_matrix(target_all[:,i].cpu(), pred_all[:,i].cpu(),
                                                title='Confusion Matrix for Degree{}'.format(i+1), normalize=True)
            plt.savefig('./saved_models/images/v2/confusion_matrix/multi_degree{}.png'.format(i+1), dpi=600)


    print("TEST_Trainning accuracy:", Accu)
    print("TEST_Trainning abs accuracy:", Accu_abs)

    TP_all_ = np.mean(TP_all)
    FP_all_ = np.mean(FP_all)
    FN_all_ = np.mean(FN_all)
    Prec = (TP_all_) / (TP_all_ + FP_all_)
    Recall = TP_all_ / (TP_all_ + FN_all_)
    F1 = (2 * TP_all_) / (2 * TP_all_ + FP_all_ + FN_all_)
    logic = int(Prec > Prec_top) + int(Recall > Recall_top) + int(F1 > F1_top) + int(Accu > Accu_top) >= 3

    if logic:
        torch.save(model.state_dict(), "./saved_models/multi_simu/weights.pt")
        torch.save(model, "./saved_models/whole_models/multi_simu.pt")

        Prec_top = Prec
        Recall_top = Recall
        F1_top = F1
        Accu_top = Accu

    return Prec_top, Recall_top, F1_top, Accu_top, Accu


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='input batch size for validation (default: 16)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='decay rate during Adam optimizing (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=500,
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--channel_p", type=int, default="21",
                        help="number of channels after projection")
    parser.add_argument("--channel_temp", type=int, default="28",
                        help="number of spatial feature extraction channel")
    parser.add_argument("--conv_pers_window", type=int, default="7",
                        help="temporal convolution window len(only odd nuber)")
    parser.add_argument("--pool_pers_window", type=int, default="3",
                        help="pooling layers window len(only odd nuber)")
    parser.add_argument("--window", type=int, default="100",
                        help="the time of a single trial")
    # parser.add_argument("--hidden_dim", type=int, default="16",
    # help="the hidden layer dimension of LSTM")
    # parser.add_argument("--embedding_dim", type=int, default="8",
    # help="the output dim of CNN embedding")
    parser.add_argument("--mode", type=str, default='multilabel_simu',
                        help="whether different trials are trained separately")
    parser.add_argument("--device", type=int, default='0',
                        help="the index of CUDA you want to use")
    parser.add_argument("--optim_scheduler", type=bool, default='1',
                        help="whether apply optimizer_learning_rate_scheduler")
    parser.add_argument("--loss_fn", type=bool, default='0',
                        help="choose between BCE(0) and Multi_Softmargin(1)")
    parser.add_argument("--threshold", type=float, default='0.5',
                        help="Threshold of classification")
    parser.add_argument("--target_dim", type=int, default='7',
                        help="the dimension of embedded sequence")

    args = parser.parse_args()

    # Configs and Hyperparameters

    batch_size = args.train_batch_size  # batch_size_trial 指外数据集录入加载数据的batch数，当训练是trial_sep
    # 模式时，每次只录入一个trial，当是mixed模式时，两个可以混同
    val_batch_size = args.val_batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    channel_p = args.channel_p
    channel_temp = args.channel_temp
    conv_pers_window = args.conv_pers_window
    pool_pers_window = args.pool_pers_window
    window = args.window  # 指一个单个数据的长度
    # hidden_dim = args.hidden_dim
    # embedding_dim = args.embedding_dim
    mode = args.mode
    log_interval = args.log_interval
    epoches = args.epochs
    device_index = args.device
    threshold = args.threshold
    target_dim = args.target_dim
    loss_fn = torch.nn.MultiLabelSoftMarginLoss() if args.loss_fn else torch.nn.BCELoss()
    drop_last = True if (mode == "multilabel_simu") else False

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_index) if use_cuda else "cpu")
    print("Pytorch Version:", torch.__version__)
    print('device={}'.format(device))
    batch_size = 32
    val_batch_size = 16
    learning_rate = 1e-4
    weight_decay = 0.001

    Prec_top, Recall_top, F1_top, Accu_top = 0.8, 0.8, 0.8, 80

    # train_dataset = DL.load_from_dir_all()
    train_dataset, test_dataset = DL.load_from_dir_all(mode, window)

    train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True, drop_last=drop_last)
    model = EMG_NET_1(channel_p=channel_p, channel_temp=channel_temp, conv_pers_window=conv_pers_window, window=window,
                      target_dim=target_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=0)

    for epoch in range(1, epoches + 1):
        train(model, device, train_loader, optimizer, epoch, loss_fn, threshold, target_dim=target_dim,
              log_interval=500)
        # val(model, device, val_loader, optimizer)
        Prec_top, Recall_top, F1_top, Accu_top, Accu = test(model, device, test_loader, threshold, target_dim=target_dim
                                                            , Prec_top=Prec_top, Recall_top=Recall_top, F1_top=F1_top,
                                                            Accu_top=Accu_top)

    save_model = False
    if (save_model):
        torch.save(model.state_dict(), "weights.pt")

    print("test training is done")
    print("\nAll train have been done！")
