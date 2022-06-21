from models.model_bina_v1 import *
from datasets import DataLoader as DL

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--degree', type=int, default=1,
                        help='the degree to be test')
    parser.add_argument('--window', type=int, default=300,
                        help='the window_length of trial')
    args = parser.parse_args()

    device = 'cuda:0'
    model = []
    test_degree = args.degree
    window = args.window
    for i in range(1, 8):
        model.append(torch.load(
            './saved_models/whole_models/degree{}.pt'.format(i),
            map_location='cuda:0'))

    test_dataset= DL.load_from_dir_all('test',degree=test_degree,window = window)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=16, shuffle=True)

    correct = 0
    TP_all, FP_all, TN_all, FN_all = 0, 0, 0, 0
    pred_all = []
    target_all = []
    output_all = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            model_ = model[test_degree - 1]
            model_.eval()
            output = model_(data)
            output = output.to(torch.float32)
            '''
            y_pred = []
            
            for i in range(7):
                model[i].eval()
                # print(signal)
                y_output = model[i](data)
                y_pred.append(y_output.argmax(dim=1, keepdim=False))        # y_pred[degree-1] 对应 degree自由度上的准确率
            '''

            # print(pred)
            # print(output.shape)
            pred = output.argmax(dim=1, keepdim=False)
            # print(pred)
            # print(pred.shape)
            # print(target.shape)
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
    print("Confusion Matrix:TP:{}  FP:{}  TN:{}  FN:{}".format(TP_all, FP_all, TN_all, FN_all))
    if (TP_all + FP_all): print('Precision:', (TP_all) / (TP_all + FP_all))
    if (TP_all + FN_all): print("TPR = Recall: ", TP_all / (TP_all + FN_all))
    if (TN_all + FP_all): print("TNR ", TN_all / (TN_all + FP_all))
    if (TN_all + FN_all): print("N_precision ", TN_all / (TN_all + FN_all))
    if (2 * TP_all + FP_all + FN_all) : print('F1:', (2 * TP_all) / (2 * TP_all + FP_all + FN_all))
    print("TEST_Trainning accuracy:", Accu)

