from __future__ import print_function
import argparse
import time

import torch.optim as optim
import wandb

import utils
from modules.masking import Masking, Masking_new
from modules.mcc import MinimumClassConfusionLoss
from modules.teacher import EMATeacher
from utils import *
from basenet import *
import os
from HybridSN import SSSE, SSSE_test, SSSE_Houston
from loss_helper import *
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics

# print(os.getcwd())
# if (os.getcwd().split('/')[-1] != 'project'):
#     os.chdir(os.path.dirname(os.getcwd()))

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=3, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--patch_size', type=int, default=7,
                    help="Size of the spatial neighbourhood (optional, if "
                         "absent will be set by the model)")
parser.add_argument('--training_sample', type=int, default=0.8,
                    help="The proportion of training source domain samples")
parser.add_argument('--training_tar_sample', type=int, default=0.3,
                    help="The proportion of training target domain samples(no labels)")
parser.add_argument('--large_num', type=int, default=3,
                    help="The data augmentation factor")
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')
# masked image consistency
parser.add_argument('--alpha', default=0.9, type=float)
parser.add_argument('--pseudo_label_weight', default="prob")
parser.add_argument('--mask_block_size', default=32, type=int)
parser.add_argument('--mask_ratio', default=0.3, type=float)
parser.add_argument('--mask_color_jitter_s', default=0.2, type=float)
parser.add_argument('--mask_color_jitter_p', default=0.2, type=float)
parser.add_argument('--mask_blur', default=True, type=bool)
parser.add_argument('--norm-mean', type=float, nargs='+',
                    default=(0.485, 0.456, 0.406), help='normalization mean')
parser.add_argument('--norm-std', type=float, nargs='+',
                    default=(0.229, 0.224, 0.225), help='normalization std')

parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                    help='Number of iterations per epoch')
parser.add_argument('--halfwidth', default=3, type=int)

# mask settings
parser.add_argument('--A_mask_ratio', default=0.1, type=float)
parser.add_argument('--spectral_A_mask', action='store_true',default=True)
parser.add_argument('--spatial_A_mask', action='store_true',default=True)
parser.add_argument('--spectral_A_block_size', default=6, type=int)
parser.add_argument('--spatial_A_block_size', default=2, type=int)
parser.add_argument('--B_mask_ratio', default=0.1, type=float)
parser.add_argument('--spectral_B_mask', action='store_true',default=True)
parser.add_argument('--spatial_B_mask', action='store_true',default=True)
parser.add_argument('--spectral_B_block_size', default=6, type=int)
parser.add_argument('--spatial_B_block_size', default=2, type=int)

# project_name
parser.add_argument('--project_sub_name', type=str,
                    default="default", help="sub name of project")
parser.add_argument('--log_results', action='store_true',
                    help="To log results in wandb")
parser.add_argument('--log_name', type=str,
                    default="log", help="log name for wandb")

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")
# mcc loss
parser.add_argument('--temperature', default=9.0,
                    type=float, help='parameter temperature scaling')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEV = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
num_epoch = args.num_epoch
batch_size = args.batch_size
lr = args.lr
num_k = args.num_k
large_num = args.large_num
use_gpu = torch.cuda.is_available()
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':
    acc_test_list1 = np.zeros([args.num_trials, args.num_trials])
    acc_test_list2 = np.zeros([args.num_trials, args.num_trials])

if args.log_results:
    os.environ['WANDB-API_KEY'] = ''
    os.environ['WANDB_MODE'] = ''
    run = wandb.init(
        project="MSDA",
        name=args.log_name)
    wandb.config.update(args)

# nDataSet = 10
num_classes = 7
N_BANDS = 48
HalfWidth = args.halfwidth
BATCH_SIZE = 32
patch_size = 2 * HalfWidth + 1

seeds = [1599,1364,1379,1672,1304,1460,1718,1603,1779,1358]

nDataSet = len(seeds)
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, num_classes])
k = np.zeros([nDataSet, 1])
# best_predict_all = []
best_acc_all = 0.0
best_predict_all = 0
best_test_acc = 0
best_source_feature = []
best_source_label = []
best_target_feature = []
best_target_label = []
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None
for iDataSet in range(nDataSet):
    data_path_s = 'data/Houston/Houston13.mat'
    label_path_s = 'data/Houston/Houston13_7gt.mat'
    data_path_t = 'data//Houston/Houston18.mat'
    label_path_t = 'data//Houston/Houston18_7gt.mat'

    data_s, label_s = utils.load_data_houston(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_houston(data_path_t, label_path_t)
    print('#######################idataset######################## ', iDataSet)
    utils.seed_everything(seeds[iDataSet], use_deterministic=True)

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, Gr, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)
    testID_s, testX_s, testY_s, Gr_s, RandPerm_s, Row_s, Column_s = utils.get_all_data(data_s, label_s, HalfWidth)

    test_source_dataset = TensorDataset(torch.tensor(testX_s), torch.tensor(testY_s))
    train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    train_tar_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    source_test_loader = DataLoader(test_source_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    len_src_loader = len(train_loader)
    len_tar_train_loader = len(train_tar_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_train_dataset = len(train_tar_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)
    ##########################################################################################################
    G = SSSE_Houston(input_channels=N_BANDS,
                     patch_size=patch_size,
                     n_classes=num_classes).to(DEV)
    F1 = ResClassifier(num_classes, training=True)
    F2 = ResClassifier(num_classes, training=True)
    # G.apply(weights_init)
    F1.apply(weights_init)
    F2.apply(weights_init)

    # EMA teacher
    F1_teacher = EMATeacher(G, F1, alpha=args.alpha, pseudo_label_weight=args.pseudo_label_weight).to(device)
    # F2_teacher = EMATeacher(G, F2, alpha=args.alpha, pseudo_label_weight=args.pseudo_label_weight).to(device)

    # mask A B
    masking_A = Masking_new(
        block_size=args.mask_block_size,
        ratio=args.A_mask_ratio,
        color_jitter_s=args.mask_color_jitter_s,
        color_jitter_p=args.mask_color_jitter_p,
        blur=args.mask_blur,
        mean=args.norm_mean,
        std=args.norm_std,
        spectral_mask=args.spectral_A_mask,
        spatial_mask=args.spatial_A_mask,
        spectral_block_size=args.spectral_A_block_size,
        spatial_block_size=args.spatial_A_block_size
    )
    masking_B = Masking_new(
        block_size=args.mask_block_size,
        ratio=args.B_mask_ratio,
        color_jitter_s=args.mask_color_jitter_s,
        color_jitter_p=args.mask_color_jitter_p,
        blur=args.mask_blur,
        mean=args.norm_mean,
        std=args.norm_std,
        spectral_mask=args.spectral_B_mask,
        spatial_mask=args.spatial_B_mask,
        spectral_block_size=args.spectral_B_block_size,
        spatial_block_size=args.spatial_B_block_size
    )
    # mcc
    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)


    lr = args.lr
    if args.cuda:
        G.cuda()
        F1.cuda()
        F2.cuda()
    if args.optimizer == 'momentum':
        optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                                weight_decay=0.0005)
    elif args.optimizer == 'adam':
        optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
    else:
        optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)


    def train(ep, train_loader, train_tar_loader):
        iter_source, iter_target = iter(train_loader), iter(train_tar_loader)
        criterion = nn.CrossEntropyLoss().cuda()
        G.train()
        F1.train()
        F2.train()
        num_iter = len_src_loader
        for batch_idx in range(1, num_iter):
            if batch_idx % len(train_tar_loader) == 0:
                iter_target = iter(train_tar_loader)
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            # label_source = label_source - 1
            if args.cuda:
                data1, target1 = data_source.cuda(), label_source.cuda()
                data2 = data_target.cuda()
            # when pretraining network source only
            eta = 1.0
            data = Variable(torch.cat((data1, data2), 0))
            target1 = Variable(target1)
            # mask_img = masking_A(data2)
            # x_t_A_masked = x_t_B_masked = x_t_C_masked = mask_img
            if args.spectral_A_mask or args.spatial_A_mask:
                x_t_A_masked = masking_A(data2)
            if args.spectral_B_mask or args.spatial_B_mask:
                x_t_B_masked = masking_B(data2)
            # generate pseudo-label
            # f1
            F1_teacher.update_weights(G, F1, ep * num_iter + batch_idx)
            F1_pseudo_label_t, F1_pseudo_prob_t = F1_teacher(data2)

            # Step A train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data)
            output1 = F1(output)
            output2 = F2(output)

            output_s1 = output1[0][:batch_size, :]
            output_s2 = output2[0][:batch_size, :]
            output_t1 = output1[0][batch_size:, :]
            output_t2 = output2[0][batch_size:, :]
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)

            # entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            # entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

            loss1 = criterion(output_s1, target1.long())
            loss2 = criterion(output_s2, target1.long())
            masking_loss_value = 0
            if args.spectral_A_mask or args.spatial_A_mask:
                # F1 MIC
                output = G(x_t_A_masked)
                y_t_A_f1_masked, _ = F1(output)
                if F1_teacher.pseudo_label_weight is not None:
                    ce = F.cross_entropy(y_t_A_f1_masked, F1_pseudo_label_t, reduction='none').float()
                    masking_loss_value += torch.mean(F1_pseudo_prob_t * ce)
                else:
                    masking_loss_value += F.cross_entropy(y_t_A_f1_masked, F1_pseudo_label_t)

            mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))
            all_loss = loss1 + loss2 + masking_loss_value * 0.5 + mcc_loss_value

            if args.log_results:
                wandb.log({f'{seeds[iDataSet]}_A_loss': all_loss,
                           f'{seeds[iDataSet]}_A_mask loss': masking_loss_value * 0.5})
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            # Step B train classifier to maximize discrepancy
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            output = G(data)
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[0][:batch_size, :]
            output_s2 = output2[0][:batch_size, :]
            output_t1 = output1[0][batch_size:, :]
            output_t2 = output2[0][batch_size:, :]
            output_t1 = F.softmax(output_t1, dim=1)
            output_t2 = F.softmax(output_t2, dim=1)
            loss1 = criterion(output_s1, target1.long())
            loss2 = criterion(output_s2, target1.long())
            # MIC
            masking_loss_value = 0
            if args.spectral_B_mask or args.spatial_B_mask:
                # F1 MIC
                output = G(x_t_B_masked)
                y_t_B_f1_masked, _ = F1(output)
                if F1_teacher.pseudo_label_weight is not None:
                    ce = F.cross_entropy(y_t_B_f1_masked, F1_pseudo_label_t, reduction='none').float()
                    masking_loss_value += torch.mean(F1_pseudo_prob_t * ce)
                else:
                    masking_loss_value += F.cross_entropy(y_t_B_f1_masked, F1_pseudo_label_t)


            loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
            # mcc_loss_value = 0.0
            mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))
            F_loss = loss1 + loss2 - eta * loss_dis + masking_loss_value * 0.5 + mcc_loss_value
            if args.log_results:
                wandb.log({f'{seeds[iDataSet]}_B_loss': F_loss,
                           f'{seeds[iDataSet]}_B_mask loss': masking_loss_value * 0.5})
            F_loss.backward()
            optimizer_f.step()
            # Step C train genrator to minimize discrepancy
            for i in range(num_k):
                optimizer_g.zero_grad()
                output = G(data)
                output1 = F1(output)
                output2 = F2(output)

                output_s1 = output1[0][:batch_size, :]
                output_s2 = output2[0][:batch_size, :]
                output_t1 = output1[0][batch_size:, :]
                output_t2 = output2[0][batch_size:, :]

                loss1 = criterion(output_s1, target1.long())
                loss2 = criterion(output_s2, target1.long())
                output_t1 = F.softmax(output_t1, dim=1)
                output_t2 = F.softmax(output_t2, dim=1)

                mcc_loss_value = (mcc_loss(output_t1) + mcc_loss(output_t2))
                loss_dis = torch.mean(torch.abs(output_t1 - output_t2)) + mcc_loss_value
                # entropy_loss = -torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
                # entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

                if args.log_results:
                    wandb.log({f'{seeds[iDataSet]}_C_loss': loss_dis,
                               f'{seeds[iDataSet]}_C_mask loss': masking_loss_value * 0.5})
                loss_dis.backward()
                optimizer_g.step()
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} MCC: {:.6f}'.format(
                        ep, batch_idx * len(data), args.batch_size * len(train_loader),
                            100. * batch_idx / len(train_loader), loss1.item(), loss2.item(), loss_dis.item(),
                        mcc_loss_value))

            if batch_idx == 1 and ep > 1:
                G.train()
                F1.train()
                F2.train()


    def test(test_loader):
        G.eval()
        F1.eval()
        F2.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        size = 0
        predict = np.array([], dtype=np.int64)
        labels = np.array([], dtype=np.int64)
        pred1_list, pred2_list, label_list, outputdata = [], [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEV), target.to(DEV)
                target2 = target
                data1, target1 = Variable(data), Variable(target2)
                output = G(data1)
                outputdata.append(output.cpu().numpy())
                output1 = F1(output)
                output2 = F2(output)
                test_loss += F.nll_loss(output1, target1.long()).item()

                pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
                correct += pred1.eq(target1.data).cpu().sum()
                pred2 = output2.data.max(1)[1]  # get the index of the max log-probability
                correct2 += pred2.eq(target1.data).cpu().sum()
                k = target1.data.size()[0]
                pred1_list.append(pred1.cpu().numpy())
                pred2_list.append(pred2.cpu().numpy())
                predict = np.append(predict, pred1.cpu().numpy())
                labels = np.append(labels, target.cpu().numpy())
                label_list.append(target2.cpu().numpy())
                size += k
                acc1 = 100. * float(correct) / float(size)
                acc2 = 100. * float(correct2) / float(size)

            test_loss = test_loss
            test_loss /= len(test_loader)  # loss function already averages over batch size
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ({:.2f}%)\n'.format(
                test_loss, correct, len_tar_dataset,
                100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset))
            # if 100. * correct / size > 67 or 100. * correct2 / size > 67:
            value = max(100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset)
            if args.log_results:
                wandb.log({f'{seeds[iDataSet]}_result_1': 100. * correct / len_tar_dataset,
                           f'{seeds[iDataSet]}_result_2': 100. * correct2 / len_tar_dataset})
        return value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata, label_list, predict, labels





    train_start = time.time()
    for ep in range(1, num_epoch + 1):
        train(ep, train_loader, train_tar_loader)
        # correct = val(val_loader)
        # 5 epoch test
        if ep % num_epoch == 0:
            test_start = time.time()
            value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata_target, target_label, predict, labels = test(
                test_loader)
            test_end = time.time()
            acc[iDataSet] = acc1
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            if acc1 >= best_test_acc:
                # torch.save(G,"checkpoints/houston/G.pt")
                # torch.save(F1, "checkpoints/houston/F1.pt")
                # torch.save(F2, "checkpoints/houston/F2.pt")
                best_test_acc = acc1
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = Gr, RandPerm, Row, Column

    train_end = time.time()


AA = np.mean(A, 1)
AAMean = np.mean(AA, 0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - test_start))
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(num_classes):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))