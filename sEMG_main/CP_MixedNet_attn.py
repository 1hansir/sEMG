# from tkinter import _Padding
import torch.optim as optim
import math
import torch.utils.data.dataloader as DataLoader
from datasets import DataLoader as DL
from argparse import ArgumentParser
from block_recurrent_transformer.transformer_attn import *
from torchtyping import TensorType

SeqTensor = TensorType['batch', 'seq_len', 'token_dim']
StateTensor = TensorType['batch', 'state_len', 'state_dim']

# constants

DEFAULT_DIM_HEAD = 32
MIN_DIM_HEAD = 16
HEAD_NUM = 8



class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, eps=1e-3, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class EMG_NET_1(nn.Module):  # Firstly attention_project different channel
    def __init__(self,channel_p=21, channel_temp=25, conv_pers_window=11, pool_pers_window=3,
                                        window=100,target_dim= 8,input_channel = 4):
        super(EMG_NET_1, self).__init__()

        self.target_dim = target_dim
        self.channel_p = channel_p
        self.channel_temp = channel_temp
        self.conv_pers_window = conv_pers_window
        self.pool_pers_window = pool_pers_window
        self.window = window
        self.input_channel = input_channel
        # The input size is (trails * channels * 1 * time_indexs)
        # ***CP-Spatio-Temporal Block***
        # Channel Projection
        self.channelProj = nn.Conv2d(self.input_channel, self.channel_p, 1, stride=1, bias=False)  # (7*1*index)->(21*1*index)
        self.batchnorm_proj_tranf = nn.BatchNorm2d(self.channel_p)
        self.batchnorm_proj_tranf_2 = nn.BatchNorm2d(self.channel_p)
        # Shape Transformation
        self.shapeTrans = nn.Conv2d(self.channel_p, self.channel_p, 1, stride=1,
                                    bias=False)  # (21*1*2700)->(21*1*2700)  ??????????????????????????????
        # Temporal Convolution

        self.drop1 = nn.Dropout2d(p=0.5)          # ?????????????????????????????????batch_size??????8????????????1/2???^8?????????????????????????????????channel?????????0(????????????batch????????????0????????????)
        self.conv1 = nn.Conv2d(1, self.channel_temp, (1, self.conv_pers_window), stride=1,
                               bias=False)  # (1*21*2700)->(25*21*2690)   #TIME-feature extraction
        self.batchnorm1 = nn.BatchNorm2d(self.channel_temp, False)
        # Spatial Convolution
        self.drop2 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(self.channel_temp, self.channel_temp, (self.channel_p, 1), stride=1, padding=0,
                               bias=False)  # (25*21*2690)->(25*2*2690)   #spatial-feature extraction
        self.batchnorm2 = nn.BatchNorm2d(self.channel_temp, False)
        # Max Pooling
        self.maxPool1 = nn.MaxPool2d((1, self.pool_pers_window), stride=(1,self.pool_pers_window),
                                     padding=0)  # (25*2*2690)->(25*2*893)

        # ***MS-Conv Block***
        # unDilated Convolution
        self.drop3 = nn.Dropout2d(p=0.5)
        # self.conv3 = nn.Conv2d(25, 50, 1, stride=1, bias=False)        # (25*2*893)->(25*1*893)
        # self.batchnorm3 = nn.BatchNorm2d(50)
        # self.drop4 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(self.channel_temp, self.channel_temp, (1, self.conv_pers_window), stride=1,
                               padding=(0, (self.conv_pers_window - 1) // 2), bias=False)  # (25*2*893)->(25*2*893)
        self.batchnorm3 = nn.BatchNorm2d(self.channel_temp)
        # Dilated Convolution
        self.dropDil = nn.Dropout2d(p=0.5)
        self.dilatedconv = nn.Conv2d(self.channel_temp, self.channel_temp, (1, self.conv_pers_window), stride=1,
                                     padding=(0, self.conv_pers_window - 1), dilation=2,
                                     bias=False)  # (25*2*893)->(25*2*893)
        self.batchnormDil = nn.BatchNorm2d(self.channel_temp)
        # Max pooling after Concatenating
        self.batchnorm_cancat = nn.BatchNorm2d(3 * self.channel_temp)
        self.poolConcatenated = nn.MaxPool2d((1, self.pool_pers_window), stride=(1,self.pool_pers_window),
                                             padding=0)  # (75*1*893)->(75*1*267)

        # ***Classification Block***
        self.drop5 = nn.Dropout(p=0.5)
        self.conv5 = nn.Conv2d(3 * self.channel_temp, 3 * self.channel_temp, (1, self.conv_pers_window),
                               stride=1)  # (75*2*267)->(75*2*257)
        self.batchnorm5 = nn.BatchNorm2d(3 * self.channel_temp)
        self.maxPool2 = nn.MaxPool2d((1, self.pool_pers_window), stride=(1,self.pool_pers_window),
                                     padding=0)  # (75*2*257)->(75*2*86)
        self.fc_dim = (((self.window - (self.conv_pers_window - 1)) // self.pool_pers_window)//self.pool_pers_window - (
                    self.conv_pers_window - 1)) // self.pool_pers_window

        # ???????????????????????????2*n????????????????????????1*n
        self.fc = nn.Linear(3 * self.channel_temp * self.fc_dim, target_dim ,bias=False)  # (1*6450)->(1*7)  ???????????????7?????????????????????7???????????????channel???7????????????7
        # self.softmax = nn.Softmax(dim=1)
        # self.batchnorm6 = nn.BatchNorm1d(7)
        # self.softmax = nn.Softmax(dim=-1)       #???????????????????????????????????????????????????-1???

        # weight initialization ????????????????????????conv??? kernel?????????????????????????????????batchnorm???weight = 1???bias = 0???
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
        # print('Channel Projection:',x)
        x = F.elu(self.batchnorm_proj_tranf_2(self.shapeTrans(x)))
        # print('before Shape Transformation:',x)
        x = torch.transpose(x, 1, 2)  # ?????????
        # print('after Shape Transformation:',x.shape)
        # x = self.drop1(x)
        # print('Temporal convolution 1:', x)
        x = self.conv1(x)
        #print('Temporal convolution 2:', x)
        x = self.batchnorm1(x)
        #print('Temporal convolution 3:', x)
        x = F.elu(x)
        # x = F.elu(self.batchnorm1(self.conv1(self.drop1(x))))
        #print('Temporal convolution 4:',x)
        x = F.elu(self.batchnorm2(self.conv2(x)))
        # print('Spatial convolution:',x.shape)
        x = self.maxPool1(x)
        # print('Max pooling???',x.shape)

        # x1 = F.elu(self.batchnorm3(self.conv3(self.drop3(x))))
        x_dilated = F.elu(self.batchnormDil(self.dilatedconv(x)))
        # print('Dilated Convolution1:', x_dilated.shape)
        x_undilated = F.elu(self.batchnorm3(self.conv3(x)))
        # print('Undilated Convolution2:', x_undilated.shape)

        x = torch.cat((x, x_dilated, x_undilated), dim=1)
        # print('Concatenated:', x.shape)

        x = self.poolConcatenated(self.batchnorm_cancat(x))
        # print('MixedScaleConv:', x.shape)

        x = F.elu(self.batchnorm5(self.conv5(x)))
        # print('Conv5:', x.shape)
        x = self.maxPool2(x)
        # print('maxPool2:', x.shape)
        x = x.view(-1, 3*self.channel_temp*self.fc_dim)
        # print('beforeFC:', x.shape)
        # print(self.fc_dim)
        x = self.fc(x)
        # print('FC:', x.shape)
        # x = self.drop5(x)          # ???????????????????????????????????????????????????????????????????????????????????????dropout
        # x = F.log_softmax(x, dim=1)      # ???????????????????????????????????????????????????loss??????MSE,????????????????????????softmax???softmax?????????????????????????????????channel??????
        # print("softmax:",x.shape)
        return x
        # ?????????????????????softmax?????????????????????????????????NLLloss()???softmax???????????????2?????????

class BRT(EMG_NET_1,nn.Module):
    def __init__(self,device, channel_p=21, channel_temp=25, conv_pers_window=11, pool_pers_window=3,
     window= 100,
    input_channel = 4,
    seq_len = 32,
    state_len = 32,
    dim_emb = 8  ,
    dim_state = 8 ,
    dim_h = 32,
    heads = 8,
    batch_size = 32,
     ):

        super(BRT, self).__init__()
        self.state_len = state_len
        self.seq_len = seq_len
        self.dim_h = dim_h
        self.dim_state = dim_state
        self.dim_emb = dim_emb
        self.batch_size = batch_size
        self.device = device

        # self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.embedding = EMG_NET_1(channel_p=channel_p, channel_temp=channel_temp,
                                         conv_pers_window=conv_pers_window
                                         , pool_pers_window=pool_pers_window, window=window, target_dim=dim_emb,
                                   input_channel= input_channel).to(self.device)

        self.BRT_atten = BlockRecurrentAttention(dim=self.dim_emb, dim_state=self.dim_state,dim_h=self.dim_h,
                                                 state_len=state_len, heads=heads,device=self.device).to(self.device)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.state_len,self.dim_state).to(self.device)

    def forward(self,
        x: SeqTensor,
        state: Optional[StateTensor] = None,           # ???????????????????????????????????????????????????????????????????????????
        mask = None,
        state_mask = None):

        # ?????????seq??????????????????embedding
        x_ite = x.unbind(dim = 2)  # ??????????????????seq_len??????????????????
        x_embed = list(map(self.embedding,x_ite))
        x_embed = [torch.unsqueeze(x,dim=1) for x in x_embed]   # ??????seq?????????

        x_embbed = torch.cat(x_embed,dim=1).to(self.device)
        output, self.hidden = self.BRT_atten(x_embbed, self.hidden,mask,state_mask)

        output = F.log_softmax(output, dim = -1)

        return output



def train(model, device, train_loader, optimizer, epoch,loss_fn,
          log_interval=100, ):  # ??????100???batch???????????????????????????????????????12800??????????????????????????????????????????????????????????????????batch_idx=0???????????????????????????
    model.train()
    correct = 0
    # loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
    # loss_fn = torch.nn.NLLLoss()
    loss_fn = loss_fn

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        output = output.to(torch.float32)
        target = target.to(torch.float32)              # ?????????BCEloss???output???target?????????????????????
        # print(output)
        # print(target.shape)
        loss = loss_fn(output, target)
        # print(loss)
        # loss = F.nll_loss(output, target.squeeze())  #target ??? shape??????1???

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
        pred = output.argmax(dim=1, keepdim=False)
        #if epoch > 3:
            #print(pred)
            #print(target.argmax(dim=1, keepdim=False))
        correct += pred.eq(target.argmax(dim=1, keepdim=False).view_as(pred)).sum().item()

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
        correct += pred.eq(target.argmax(dim=1, keepdim=False).view_as(pred)).sum().item()  # view_as reshape the [target] and eq().sum().item()  get the sum of the correct validation
        if batch_idx == 0:
            print("pred:", output[0])
            print("true:", target[0])
    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def test(model, device, test_loader):  # ??????100???batch???????????????????????????????????????12800??????????????????????????????????????????????????????????????????batch_idx=0???????????????????????????
    model.eval()
    correct = 0

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
            correct += pred.eq(target.argmax(dim=1, keepdim=False).view_as(pred)).sum().item()

    print("TEST_Trainning accuracy:", 100. * correct / (len(test_loader.dataset)))

'''
def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(logdir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
'''

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
    parser.add_argument("--channel_temp", type=int, default="32",
                        help="number of spatial feature extraction channel")
    parser.add_argument("--conv_pers_window", type=int, default="7",
                        help="temporal convolution window len(only odd nuber)")
    parser.add_argument("--pool_pers_window", type=int, default="3",
                        help="pooling layers window len(only odd nuber)")
    parser.add_argument("--window", type=int, default="100",
                        help="the time of a single trial")
    parser.add_argument("--input_channel", type=int, default="4",
                       help="the input channel num")
    parser.add_argument("--mode", type=str, default='CNN_multiclass',
                        help="whether different trials are trained separately")
    parser.add_argument("--device", type=int, default='0',
                        help="the index of CUDA you want to use")
    parser.add_argument("--optim_scheduler", type=bool, default='1',
                        help="whether apply optimizer_learning_rate_scheduler")
    parser.add_argument("--loss_fn", type=bool, default='0',
                        help="choose between BCE(0) and Multi_Softmargin(1)")

    parser.add_argument("--seq_len", type=int, default='20',
                        help="The length of seq(and hidden state)")
    parser.add_argument("--dim_emb", type=int, default='8',
                        help="The dimension of CNN's embedding")
    parser.add_argument("--dim_state", type=int, default='8',
                        help="The dimension of Transformer's hidden state")
    parser.add_argument("--dim_h", type=int, default='8',
                        help="The dimension of the middle output of a single head")
    parser.add_argument("--head", type=int, default='4',
                        help="The number of layer")

    args = parser.parse_args()

    # Configs and Hyperparameters

    batch_size = args.train_batch_size  # batch_size_trial ????????????????????????????????????batch??????????????????trial_sep
    # ?????????????????????????????????trial?????????mixed??????????????????????????????
    val_batch_size = args.val_batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    channel_p = args.channel_p
    channel_temp = args.channel_temp
    conv_pers_window = args.conv_pers_window
    pool_pers_window = args.pool_pers_window
    window = args.window  # ??????????????????????????????
    input_channel = args.input_channel
    mode = args.mode
    log_interval = args.log_interval
    epoches = args.epochs
    device_index = args.device

    state_len = seq_len = args.seq_len
    dim_emb = args.dim_emb  # ??????debug???????????????8
    dim_state = args.dim_state
    dim_h = args.dim_h
    heads = args.heads

    loss_fn = torch.nn.MultiLabelSoftMarginLoss() if args.loss_fn else torch.nn.BCELoss()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_index) if use_cuda else "cpu")
    print("Pytorch Version:", torch.__version__)
    print('device={}'.format(device))
    batch_size = 64
    val_batch_size = 16
    learning_rate = 1e-4
    weight_decay = 0.001

    # train_dataset = DL.load_from_dir_all()
    train_dataset,test_dataset = DL.load_from_dir_all(mode,window)

    train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True)
    model = BRT(device,channel_p=channel_p,channel_temp=channel_temp,conv_pers_window=conv_pers_window,window=window,
                        input_channel=input_channel, dim_emb=dim_emb, dim_state=dim_state, dim_h=dim_h,
                        state_len=state_len,heads=heads,batch_size=batch_size).to(device)


    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=0)


    for epoch in range(1, epoches + 1):
        train(model, device, train_loader, optimizer, epoch,loss_fn,log_interval=500)
        # val(model, device, val_loader, optimizer)
        test(model, device, test_loader)

    save_model = False
    if (save_model):
        torch.save(model.state_dict(), "weights.pt")

    '''
    for sub in range(1,9):
        #train_dataset = DL.import_EEGData(0, 1, 'data/A0')
        X = np.load("processed_data/train_data_BCI_{}.npy".format(sub))
        y = np.load("processed_data/train_label_BCI_{}.npy".format(sub))
        y = y[:, np.newaxis]
        print(X.shape, y.shape)
        train_dataset = DL.EEGDataset(X, y)

        X = np.load("processed_data/val_data_BCI_{}.npy".format(sub))
        y = np.load("processed_data/val_label_BCI_{}.npy".format(sub))
        y = y[:, np.newaxis]
        print("shape of val set:", X.shape, y.shape)
        val_dataset = DL.EEGDataset(X, y)

        X = np.load("processed_data/test_data_BCI_{}.npy".format(sub))
        y = np.load("processed_data/test_label_BCI_{}.npy".format(sub))
        y = y[:, np.newaxis]
        print("shape of test set:", X.shape, y.shape)
        test_dataset = DL.EEGDataset(X, y)


        train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

        model = EMG_NET_1().to(device)
        #writer = SummaryWriter('tensorboard_logs')



        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        epoches = 500
        for epoch in range(1, epoches + 1):
            train(model, device, train_loader, optimizer, epoch)
            val(model, device, val_loader, optimizer)
            test(model, device, test_loader)
        save_model = False
        if (save_model):
            torch.save(model.state_dict(), "weights.pt")

        print("-----------------------")
        print("The training of {}th subject is complete!".format(sub))
        print("-----------------------")
        '''

    print("test training is done")
    print("\nAll train have been done???")
