import numpy as np
import torch
from numpy.random import *
from torch.autograd import Variable
from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm1d, BatchNorm3d
import scipy.io as sio
import os
import random
from sklearn import preprocessing

def get_sample_data(Sample_data, Sample_label, HalfWidth, num_per_class):
    print('get_sample_data() run...')
    print('The original sample data shape:',Sample_data.shape)
    nBand = Sample_data.shape[2]

    data = np.pad(Sample_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(Sample_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    m = int(np.max(label))
    print(f'num_class : {m}')

    val = {}
    val_indices = []

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices[:num_per_class]
        val[i] = indices[num_per_class:]

    for i in range(m):
        train_indices += train[i]
        val_indices += val[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    #val
    print('the number of val data:', len(val_indices))
    nVAL = len(val_indices)
    val_data = np.zeros([nVAL, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    val_label = np.zeros([nVAL], dtype=np.int64)
    RandPerm = val_indices
    RandPerm = np.array(RandPerm)

    for i in range(nVAL):
        val_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        val_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    val_label = val_label - 1

    #train
    print('the number of processed data:', len(train_indices))
    nTrain = len(train_indices)
    index = np.zeros([nTrain], dtype=np.int64)
    processed_data = np.zeros([nTrain, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTrain], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTrain):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('sample data shape', processed_data.shape)
    print('sample label shape', processed_label.shape)
    print('get_sample_data() end...')
    return processed_data, processed_label#, val_data, val_label

def get_all_data(All_data, All_label, HalfWidth):
    print('get_all_data() run...')
    print('The original data shape:', All_data.shape)
    nBand = All_data.shape[2]

    data = np.pad(All_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    label = np.pad(All_label, HalfWidth, mode='constant')

    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    num_class = int(np.max(label))
    print(f'num_class : {num_class}')

    for i in range(num_class):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   label[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        train[i] = indices

    for i in range(num_class):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of all data:', len(train_indices))
    nTest = len(train_indices)
    index = np.zeros([nTest], dtype=np.int64)
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTest):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                          Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :],
                                          (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')
    return index, processed_data, processed_label, label, RandPerm, Row, Column


def get_all_test_data(All_data, All_label, HalfWidth):
    print('get_all_data() run...')
    print('The original data shape:', All_data.shape)
    nBand = All_data.shape[2]

    data = np.pad(All_data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode='constant')
    All_label += 1
    label = np.pad(All_label, HalfWidth, mode='constant')
    # label += 1
    train = {}
    train_indices = []
    [Row, Column] = np.nonzero(label)
    num_class = int(np.max(label))
    print(f'num_class : {num_class}')

    for i in range(num_class):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if
                   (label[Row[j], Column[j]] == i + 1 and not np.all(data[Row[j]][Column[j]] == 0))]
        np.random.shuffle(indices)
        train[i] = indices

    for i in range(num_class):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of all data:', len(train_indices))
    nTest = len(train_indices)
    index = np.zeros([nTest], dtype=np.int64)
    processed_data = np.zeros([nTest, nBand, 2 * HalfWidth + 1, 2 * HalfWidth + 1], dtype=np.float32)
    processed_label = np.zeros([nTest], dtype=np.int64)
    RandPerm = train_indices
    RandPerm = np.array(RandPerm)

    for i in range(nTest):
        index[i] = i
        processed_data[i, :, :, :] = np.transpose(data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, \
                                                  Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1,
                                                  :],
                                                  (2, 0, 1))
        processed_label[i] = label[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    processed_label = processed_label - 1

    print('processed all data shape:', processed_data.shape)
    print('processed all label shape:', processed_label.shape)
    print('get_all_data() end...')
    All_label -= 1
    label_test = np.pad(All_label, HalfWidth, mode='constant')
    return index, processed_data, processed_label, label_test, RandPerm, Row, Column


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
def seed_everything(seed,use_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    if use_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def cubeData(file_path):
    total = sio.loadmat(file_path)

    data1 = total['DataCube1'] #up
    data2 = total['DataCube2'] #pc
    gt1 = total['gt1']
    gt2 = total['gt2']

    # Data_Band_Scaler_s = data1
    # Data_Band_Scaler_t = data2
    # print('max and min ')

    # 归一化 [-0.5,0.5]
    # data1 = data1.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler_s = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))# - 0.5
    #
    # data2 = data2.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler_t = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2)) #- 0.5

    # # # 标准化
    data_s = data1.reshape(np.prod(data1.shape[:2]), np.prod(data1.shape[2:]))  # (111104,204)
    data_scaler_s = preprocessing.scale(data_s)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_s = data_scaler_s.reshape(data1.shape[0], data1.shape[1],data1.shape[2])

    data_t = data2.reshape(np.prod(data2.shape[:2]), np.prod(data2.shape[2:]))  # (111104,204)
    data_scaler_t = preprocessing.scale(data_t)  #标准化 (X-X_mean)/X_std,
    Data_Band_Scaler_t = data_scaler_t.reshape(data2.shape[0], data2.shape[1],data2.shape[2])
    print(np.max(Data_Band_Scaler_s),np.min(Data_Band_Scaler_s))
    print(np.max(Data_Band_Scaler_t),np.min(Data_Band_Scaler_t))
    return Data_Band_Scaler_s,Data_Band_Scaler_t, gt1,gt2  # image:(512,217,3),label:(512,217)

def load_data_hyrank(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['ori_data']

    GroundTruth = label_data['map']

    # Data_Band_Scaler = data_all


    # # 归一化
    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def load_data_pavia(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)


    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # Data_Band_Scaler = (data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all))

    # Data_Band_Scaler = data_all

    print(np.max(Data_Band_Scaler),np.min(Data_Band_Scaler))

    return Data_Band_Scaler, GroundTruth  # image:(512,217,3),label:(512,217)

def load_data_houston13(image_file, train_label_file,test_label_file):
    image_data = sio.loadmat(image_file)
    train_label_data = sio.loadmat(train_label_file)
    test_label_data = sio.loadmat(test_label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['data']

    GroundTruth_train = train_label_data['mask_train']

    GroundTruth_test = test_label_data['mask_test']

    # Data_Band_Scaler = data_all


    # # 归一化
    # data_all = data_all.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth_train,GroundTruth_test # image:(512,217,3),label:(512,217)

def load_data_houston(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['ori_data']

    GroundTruth = label_data['map']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)

def load_data_YRD(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    # print(image_data.keys()) #mine
    # print(label_data.keys())

    data_all = image_data['HSI']

    GroundTruth = label_data['GT']

    Data_Band_Scaler = data_all


    # # 归一化
    # data = data.astype(np.float32)  # 半精度浮点：1位符号，5位指数，10位尾数
    # data_all = 1 * ((data_all - np.min(data_all)) / (np.max(data_all) - np.min(data_all)) - 0.5)

    # data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    # Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    print(np.max(Data_Band_Scaler), np.min(Data_Band_Scaler))
    return Data_Band_Scaler, GroundTruth # image:(512,217,3),label:(512,217)


def textread(path):
    # if not os.path.exists(path):
    #     print path, 'does not exist.'
    #     return False
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
    return lines

def adjust_learning_rate(optimizer, epoch,lr=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * 0.99#min(1, 2 - epoch/float(20))#0.95 best
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)

def cdd(output_t1,output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss

