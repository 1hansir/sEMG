# from tkinter import _Padding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tensorboardX import SummaryWriter
import numpy as np
import torch.utils.data.dataloader as DataLoader
from datasets import DataLoader as DL
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import scikitplot as skplt



class EMG_NET_1(nn.Module):
    def __init__(self, channel_p=21, channel_temp=25, conv_pers_window=11, pool_pers_window=3, conv_degree_window=7
                 , pool_pers_degree=3, window=100, target_dim=2):
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
                                     padding=0)  # (25*2*2690)->(25*2*893)

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
        self.poolConcatenated = nn.MaxPool2d((self.pool_pers_degree, 1), stride=(self.pool_pers_degree // 2, 1),
                                             padding=0)  # (75*1*893)->(75*1*267)

        # ***Classification Block***

        self.fc_dim = (((3 * self.channel_temp - (self.conv_pers_degree - 1)) - self.pool_pers_degree) //
                       (self.pool_pers_degree // 2) + 1) * \
                      (((self.window - (self.conv_pers_window - 1)) // self.pool_pers_window) - (self.conv_pers_window - 1))

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
        # print("input size:",x.shape)
        x = F.elu(self.batchnorm_proj_tranf(self.channelProj(x)))
        # print('Channel Projection:',x.shape)
        x = F.elu(self.batchnorm_proj_tranf_2(self.shapeTrans(x)))
        # print('before Shape Transformation:',x.shape)
        x = torch.transpose(x, 1, 2)  # 交换轴
        # print('after Shape Transformation:',x.shape)

        # print('Temporal convolution 1:', x.shape)
        x = self.conv1(x)
        # print('Temporal convolution 2:', x.shape)
        x = self.batchnorm1(x)
        # print('Temporal convolution 3:', x.shape)
        x = F.elu(x)

        # print('Temporal convolution 4:',x.shape)
        x = F.elu(self.batchnorm2(self.conv2(x)))
        # print('Spatial convolution:',x.shape)
        x = self.maxPool1(x)
        # print('Max pooling：',x.shape)

        x_dilated = F.elu(self.batchnormDil(self.dilatedconv(x)))
        # print('Dilated Convolution1:', x_dilated.shape)
        x_undilated = F.elu(self.batchnorm3(self.conv3(x)))
        # print('Undilated Convolution2:', x_undilated.shape)

        x = torch.cat((x, x_dilated, x_undilated), dim=1)
        # print('Concatenated:', x.shape)
        x = torch.transpose(x, 1, 2)  # 交换轴
        # print('Transpose after Concatenated:',x.shape)
        x = self.conv_cat_degree(x)
        # print('Conv across degree and time',x.shape)

        x = F.elu(self.poolConcatenated(self.batchnorm_cancat(x)))

        # print('maxPool2:', x.shape)
        x = x.view(-1, self.fc_dim)
        # print('beforeFC:', x.shape)
        x = self.fc(x)

        return x



def binary_confusion_matrix(acts, pres):  # 混淆矩阵代码
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(acts)):
        if acts[i] == 1 and pres[i] == 1:
            TP += 1
        if acts[i] == 0 and pres[i] == 1:
            FP += 1
        if acts[i] == 1 and pres[i] == 0:
            FN += 1
        if acts[i] == 0 and pres[i] == 0:
            TN += 1
    return TP, FP, TN, FN


def train(model, device, train_loader, optimizer, epoch, weight,
          log_interval=100, ):  # 每过100个batch输出一次观察，这样至少需要12800个数据，但并不存在如此多数据，因此一般只有在batch_idx=0时才会输出一次观察
    model.train()
    correct = 0
    # loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
    weight = np.array(weight)
    weight = weight.astype(np.float32)
    weight = torch.from_numpy(weight).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    loss_all = 0
    # TP_all, FP_all, TN_all, FN_all = 0, 0, 0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # output = output.to(torch.float32)
        # target = target.to(torch.float32)
        # print('output:',output)
        # print(target.shape)
        loss = loss_fn(output, target)  # 注意！crossentropy & nllloss的标签都是从0开始
        loss_all += loss

        loss.backward()
        optimizer.step()
        '''
        for i,(name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name, param, 0)
                writer.add_scalar('loss', loss, i)
                loss = loss * 0.5
        '''
        pred = output.argmax(dim=1, keepdim=False)
        # print('pred:',pred)
        # print('target:',target)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss:{:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()
            ))
            '''
            writer.add_scalar(
                "Training loss",
                loss.item(),
                epoch * len(train_loader)
            )
            '''
    print("Trainning accuracy:", 100. * correct / (len(train_loader.dataset)))
    return loss_all / len(train_loader.dataset) * 32


def val(model, device, val_loader, optimizer):
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
        pred = output.argmax(dim=1, keepdim=True)
        # correct = output.eq(target.view_as(pred)).sum().item()
        correct += pred.eq(target.view_as(
            pred)).sum().item()  # view_as reshape the [target] and eq().sum().item()  get the sum of the correct validation
        if batch_idx == 0:
            print("pred:", output[0])
            print("true:", target[0])
    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return print({'Total': total_num, 'Trainable': trainable_num})

def test(model, device, test_loader, degree, Prec_top, Recall_top, F1_top,
         Accu_top):  # 每过100个batch输出一次观察，这样至少需要12800个数据，但并不存在如此多数据，因此一般只有在batch_idx=0时才会输出一次观察
    model.eval()
    correct = 0
    TP_all, FP_all, TN_all, FN_all = 0, 0, 0, 0
    pred_all = []
    target_all = []
    output_all = []

    get_parameter_number(model)

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
            pred = output.argmax(dim=1, keepdim=False)
            # print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

            TP, FP, TN, FN = binary_confusion_matrix(target, pred)
            TP_all += TP
            FP_all += FP
            TN_all += TN
            FN_all += FN

            if batch_idx == 0:
                pred_all = pred
                target_all = target
                output_all = torch.softmax(output, dim=1)
            else:
                pred_all = torch.cat((pred_all, pred), dim=0)
                target_all = torch.cat((target_all, target), dim=0)
                output_all = torch.cat((output_all, torch.softmax(output, dim=1)), dim=0)

    Accu = 100. * correct / (len(test_loader.dataset))

    print("Confusion Matrix:", TP_all, FP_all, TN_all, FN_all)
    print('Precision:', (TP_all) / (TP_all + FP_all))
    print("TPR = Recall: ", TP_all / (TP_all + FN_all))
    print('F1:', (2 * TP_all) / (2 * TP_all + FP_all + FN_all))
    print("TEST_Trainning accuracy:", Accu)

    # 对最好结果进行储存
    Prec = (TP_all) / (TP_all + FP_all)
    Recall = TP_all / (TP_all + FN_all)
    F1 = (2 * TP_all) / (2 * TP_all + FP_all + FN_all)
    logic = int(Prec > Prec_top) + int(Recall > Recall_top) + int(F1 > F1_top) + int(Accu > Accu_top) >= 3
    if logic:
        # save the weights of model
        torch.save(model.state_dict(), "./saved_models/bina_simu/degree{}_v2/weights.pt".format(degree))
        torch.save(model, "./saved_models/whole_models/degree{}_v2.pt".format(degree))

        # plot the confusion matrix of model
        skplt.metrics.plot_confusion_matrix(target_all.cpu(), pred_all.cpu(),
                                            title='Confusion Matrix for Degree{}'.format(degree), normalize=True)
        plt.savefig('./saved_models/images/v2/confusion_matrix/degree{}.png'.format(degree), dpi=600)

        # plot the ROC curve
        skplt.metrics.plot_roc(target_all.cpu(), output_all.cpu(), title='ROC curve for degree{}'.format(degree))
        plt.savefig('./saved_models/images/v2/ROC_curve/degree{}.png'.format(degree), dpi=600)


        Prec_top = Prec
        Recall_top = Recall
        F1_top = F1
        Accu_top = Accu

    return Prec_top, Recall_top, F1_top, Accu_top,Accu


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))


def read_weight(degree):
    file = open('weight_dict_simu.txt', 'r')
    js = file.read()
    weight_dict = json.loads(js)
    weight = weight_dict['{}'.format(degree)]
    print('weight:', weight)
    file.close()

    return weight


if __name__ == "__main__":

    # Configs and Hyperparameters
    parser = ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='input batch size for validation (default: 64)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='decay rate during Adam optimizing (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=500,
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--channel_p", type=int, default="21",
                        help="number of channels after projection")
    parser.add_argument("--channel_temp", type=int, default="32",
                        help="number of spatial feature extraction channel")
    parser.add_argument("--conv_pers_window", type=int, default="7",
                        help="temporal convolution window len(only odd nuber)")
    parser.add_argument("--window", type=int, default="100",
                        help="the time of a single trial")
    # parser.add_argument("--hidden_dim", type=int, default="16",
    # help="the hidden layer dimension of LSTM")
    # parser.add_argument("--embedding_dim", type=int, default="8",
    # help="the output dim of CNN embedding")
    parser.add_argument("--mode", type=str, default='Binary_simu_classification',
                        help="whether different trials are trained separately")
    parser.add_argument("--device", type=int, default='0',
                        help="the index of CUDA you want to use")
    parser.add_argument("--optim_scheduler", type=bool, default='1',
                        help="whether apply optimizer_learning_rate_scheduler")
    parser.add_argument("--pool_pers_window", type=int, default="3",
                        help="pooling layers window len(only odd nuber)")
    parser.add_argument('--degree', type=int, default=3,
                        help='the degree that is moving', choices=[1, 2, 3, 4, 5, 6, 7])

    args = parser.parse_args()

    # Configs and Hyperparameters

    batch_size_trial = batch_size = args.train_batch_size  # batch_size_trial 指外数据集录入加载数据的batch数，当训练是trial_sep
    # 模式时，每次只录入一个trial，当是mixed模式时，两个可以混同
    val_batch_size = args.val_batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    channel_p = args.channel_p
    channel_temp = args.channel_temp
    conv_pers_window = args.conv_pers_window
    pool_pers_window = args.pool_pers_window
    window = args.window  # 指一个单个数据的长度
    degree = args.degree

    # hidden_dim = args.hidden_dim
    # embedding_dim = args.embedding_dim
    mode = args.mode
    log_interval = args.log_interval
    epoches = args.epochs
    device_index = args.device

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_index) if use_cuda else "cpu")
    print("Pytorch Version:", torch.__version__)
    print('device={}'.format(device))
    print('Degree = ', degree)

    loss_list = []
    accu_list = [0,]
    Prec_top, Recall_top, F1_top, Accu_top = 0.5, 0.5, 0.5, 50

    # train_dataset = DL.load_from_dir_all()
    train_dataset, test_dataset = DL.load_from_dir_all(mode, window, degree)
    weight = read_weight(degree)

    train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True)
    model = EMG_NET_1(channel_p=channel_p, channel_temp=channel_temp, conv_pers_window=conv_pers_window,
                      window=window).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=0)

    for epoch in range(1, epoches + 1):
        loss_list.append(train(model, device, train_loader, optimizer, epoch, weight, log_interval=500).cpu().item())
        # val(model, device, val_loader, optimizer)
        Prec_top, Recall_top, F1_top, Accu_top,Accu = test(model, device, test_loader, degree, Prec_top, Recall_top, F1_top,
                                                      Accu_top)
        accu_list.append(Accu)

    # print the learning_curve
    f1 = plt.figure()
    x1 = np.linspace(0,epoches,epoches+1)
    x2 = np.linspace(1, epoches, epoches)

    plt.subplot(211)
    plt.plot(x1, accu_list, 'r')
    plt.title('Accuracy_Curve')  # 折线图标题
    plt.xlabel('Epoches')  # x轴标题
    plt.ylabel('Accuracy')  # y轴标题
    plt.subplot(212)
    plt.plot(x2, loss_list, 'b')
    plt.title('Loss_Curve')  # 折线图标题
    plt.xlabel('Epoches')  # x轴标题
    plt.ylabel('Loss')  # y轴标题

    plt.savefig("./saved_models/images/v2/learning_curve/degree{}.png".format(degree))

    save_model = False
    if (save_model):
        torch.save(model.state_dict(), "weights.pt")


    print("test training is done")
    print("\nAll train have been done！")
