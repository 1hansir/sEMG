from models.model_multi_v1 import *
from datasets import DataLoader as DL


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return print({'Total': total_num, 'Trainable': trainable_num})

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--degree', type=int, default=1,
                        help='the degree to be test')
    parser.add_argument('--window', type=int, default=100,
                        help='the window_length of trial')
    parser.add_argument('--target_dim', type=int, default=7,
                        help='the target_dim of classification')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='the threshold of classification')
    args = parser.parse_args()

    device = 'cuda:0'
    model = []
    test_degree = args.degree
    window = args.window
    target_dim = args.target_dim
    threshold = args.threshold
    for i in range(1, 8):
        model.append(torch.load(
            './saved_models/whole_models/multi_simu.pt'.format(i),
            map_location='cuda:0'))
    get_parameter_number(model[0])

    test_dataset= DL.load_from_dir_all('test_multi',degree=test_degree,window = window)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=16, shuffle=True,drop_last=True)

    correct = 0
    correct_1 = 0
    abs = 0
    all_abs = 0
    TP_all = [0 for i in range(0, target_dim)]
    FP_all = [0 for i in range(0, target_dim)]
    FN_all = [0 for i in range(0, target_dim)]
    TN_all = [0 for i in range(0, target_dim)]

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            model_ = model[test_degree - 1]


            model_.eval()
            output = model_(data)
            output = output.to(torch.float32)


            pred = np.where(output.cpu() > threshold, 1, 0)

            pred = torch.from_numpy(pred).to(device)
            correct += pred.eq(target.view_as(
                pred)).sum().item()  # view_as reshape the [target] and eq().sum().item()  get the sum of the correct validation
            correct_1 += (pred * target.view_as(pred)).sum().item()
            all_abs += target.sum().item()


            TP, FP, TN, FN = binary_confusion_matrix(target, pred, target_dim)
            for i in range(target_dim):
                TP_all[i] += TP[i]
                FP_all[i] += FP[i]
                TN_all[i] += TN[i]
                FN_all[i] += FN[i]

    Accu_abs = 100. * correct_1 / all_abs
    Accu = 100. * correct / (len(test_loader.dataset) * target_dim)
    for i in range(target_dim):
        print('DEGREE:{}--------------------------- '.format(i+1))
        print("Confusion Matrix:TP:{}  FP:{}  TN:{}  FN:{}".format(TP_all[i], FP_all[i], TN_all[i], FN_all[i]))
        if (TP_all[i] + FP_all[i]): print('Precision:', (TP_all[i]) / (TP_all[i] + FP_all[i]))
        if (TP_all[i] + FN_all[i]): print("TPR = Recall: ", TP_all[i] / (TP_all[i] + FN_all[i]))
        if (2 * TP_all[i] + FP_all[i] + FN_all[i]):
            print('F1:', (2 * TP_all[i]) / (2 * TP_all[i] + FP_all[i] + FN_all[i]))

    print("TEST_Trainning accuracy:", Accu)
    print("TEST_Trainning abs accuracy:", Accu_abs)

