import torch
import torch.nn as nn
import torch.nn.functional as F
windowSize = 25
K = 30  # 参考Hybrid-Spectral-Net
rate = 16


class HybridSN(nn.Module):
    # 定义各个层的部分
    def __init__(self):
        super(HybridSN, self).__init__()
        self.S = windowSize
        self.L = K

        # self.conv_block = nn.Sequential()
        ## convolutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))

        # 不懂 inputX经过三重3d卷积的大小
        inputX = self.get2Dinput()
        inputConv4 = inputX.shape[1] * inputX.shape[2]
        # conv4 （24*24=576, 19, 19），64个 3x3 的卷积核 ==>（（64, 17, 17）
        self.conv4 = nn.Conv2d(inputConv4, 64, kernel_size=(3, 3))

        # self-attention
        self.sa1 = nn.Conv2d(64, 64 // rate, kernel_size=1)
        self.sa2 = nn.Conv2d(64 // rate, 64, kernel_size=1)

        # 全连接层（256个节点） # 64 * 17 * 17 = 18496
        self.dense1 = nn.Linear(18496, 256)
        # 全连接层（128个节点）
        self.dense2 = nn.Linear(256, 128)
        # 最终输出层(16个节点)
        # self.dense3 = nn.Linear(128, class_num)

        # 让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        # 参考: https://blog.csdn.net/yangfengling1023/article/details/82911306
        # self.drop = nn.Dropout(p = 0.4)
        # 改成0.43试试
        self.drop = nn.Dropout(p=0.43)
        self.soft = nn.Softmax(dim=1)
        pass

    # 辅助函数，没怎么懂，求经历过三重卷积后二维的一个大小
    def get2Dinput(self):
        # torch.no_grad(): 做运算，但不计入梯度记录
        with torch.no_grad():
            x = torch.zeros((1, 1, self.L, self.S, self.S))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x
        pass

    # 必须重载的部分，X代表输入
    def forward(self, x):
        # F在上文有定义torch.nn.functional，是已定义好的一组名称
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)

        # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
        out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
        out = self.conv4(out)
        out = F.relu(out)

        # Squeeze 第三维卷成1了
        weight = F.avg_pool2d(out, out.size(2))  # 参数为输入，kernel
        # 参考: https://blog.csdn.net/qq_21210467/article/details/81415300
        # 参考: https://blog.csdn.net/u013066730/article/details/102553073

        # Excitation: sa（压缩到16分之一）--Relu--fc（激到之前维度）--Sigmoid（保证输出为0至1之间）
        weight = F.relu(self.sa1(weight))
        weight = F.sigmoid(self.sa2(weight))
        out = out * weight

        # flatten: 变为 18496 维的向量，
        out = out.view(out.size(0), -1)

        out = F.relu(self.dense1(out))
        out = self.drop(out)
        out = F.relu(self.dense2(out))
        out = self.drop(out)
        out = self.dense3(out)

        # 添加此语句后出现LOSS不下降的情况，参考：https://www.e-learn.cn/topic/3733809
        # 原因是CrossEntropyLoss()=softmax+负对数损失（已经包含了softmax)。如果多写一次softmax，则结果会发生错误
        # out = self.soft(out)
        # out = F.log_softmax(out)

        return out


class SSSE(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='replicate',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='replicate',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128,128,self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='replicate', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='replicate', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        # Convolution layer 1
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        e2 = self.SpatialSE(x2)
        x2 = torch.mul(e2,x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x

class SSSE_test(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE_test, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128,128,self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='replicate',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='replicate', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='replicate', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        # self.avgpool = nn.Conv2d(in_channels=self.inter_size,out_channels=self.inter_size,kernel_size=self.sz)

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        # Convolution layer 1
        x = x.unsqueeze(1)
        # spatial
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        # spatial
        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        e2 = self.SpatialSE(x2)
        x2 = torch.mul(e2,x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x




class SSSE_Houston(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(SSSE_Houston, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), padding_mode='zeros',
                               bias=True)
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()
        if patch_size == 7:
            self.SpectralSE = SpectralSE(128, 128, self.sz)
        elif patch_size == 3:
            self.SpectralSE = SpectralSE_three(128, 128, self.sz)
        elif patch_size == 5:
            self.SpectralSE = SpectralSE_five(128, 128, self.sz)
        elif patch_size == 9:
            self.SpectralSE = SpectralSE_nine(128,128,self.sz)
        elif patch_size == 11:
            self.SpectralSE = SpectralSE_eleven(128, 128, self.sz)
        elif patch_size == 13:
            self.SpectralSE = SpectralSE_thirteen(128, 128, self.sz)
        else:
            raise Exception('this patch not define')
        # self.SpectralSE = SpectralSE_R(128, 128, self.sz)
        # self.SpectralSE = SpectralSE_S(128, 128, self.sz)

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros',
                               bias=True)
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), padding_mode='zeros',
                               bias=True)
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.SpatialSE = SpatialSE(24, 1)
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               padding_mode='zeros', bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                padding_mode='zeros', bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        # self.avgpool = nn.Conv2d(in_channels=self.inter_size,out_channels=self.inter_size,kernel_size=self.sz)

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, bounds=None):
        # Convolution layer 1
        x = x.unsqueeze(1)
        # spatial
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        e1 = self.SpectralSE(x1)
        x1 = torch.mul(e1, x1)

        # spatial
        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))
        e2 = self.SpatialSE(x2)
        x2 = torch.mul(e2,x2)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)

        x = self.avgpool(x)
        x = x.reshape((x.size(0), -1))

        # Fully connected layer
        # x = self.fc1(x)

        return x




class SpatialSE(nn.Module):
    # 定义各个层的部分
    def __init__(self,in_channel,out_channel):
        super(SpatialSE, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        C = x.shape[1]
        x = self.conv(x)
        x = torch.sigmoid(x)
        out = x.repeat(1,C,1,1)
        return out
# patch size 7
class SpectralSE(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

# patch_size = 5
class SpectralSE_five(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_five, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

# patch_size = 9
class SpectralSE_nine(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_nine, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=3)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

# patch_size = 11
class SpectralSE_eleven(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_eleven, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=4)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out

class SpectralSE_thirteen(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_thirteen, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=4)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=4)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out


# patch_size = 3
class SpectralSE_three(nn.Module):
    # 定义各个层的部分
    def __init__(self, in_channel, C, sz):
        super(SpectralSE_three, self).__init__()
        # 全局池化
        self.avgpool = nn.AvgPool2d((sz, sz))
        self.conv1 = nn.Conv2d(in_channel, C//4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(C//4, C, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        out = torch.sigmoid(x)
        return out