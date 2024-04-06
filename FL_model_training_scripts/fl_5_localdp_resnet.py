import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, utils, datasets, models
from torchsummary import summary
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from earlystopping import EarlyStopping

import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd

from opacus.utils.batch_memory_manager import BatchMemoryManager

from tqdm.auto import tqdm

# set manual seed for reproducibility
# seed = 42

# general reproducibility
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# DATA LOADING

fl_data = pd.read_pickle('datasets_fl/fl_data_bin_10.pickle')
agsx = pd.read_pickle('datasets_fl/agsx_df.pickle')

# MODEL
def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
class ResidualUnit(nn.Module):
    """Residual unit block (unidimensional).
    Parameters
    ----------
    n_samples_out: int
        Number of output samples.
    n_filters_out: int
        Number of output filters.
    kernel_initializer: str, optional
        Initializer for the weights matrices. See Keras initializers. By default it uses
        'he_normal'.
    dropout_keep_prob: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. Default is 17.
    preactivation: bool, optional
        When preactivation is true use full preactivation architecture proposed
        in [1]. Otherwise, use architecture proposed in the original ResNet
        paper [2]. By default it is true.
    postactivation_bn: bool, optional
        Defines if you use batch normalization before or after the activation layer (there
        seems to be some advantages in some cases:
        https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md).
        If true, the batch normalization is used before the activation
        function, otherwise the activation comes first, as it is usually done.
        By default it is false.
    activation_function: string, optional
        Keras activation function to be used. By default 'relu'.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027 [cs], Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(
        self,
        n_samples_out, 
        n_filters_out, 
        n_filters_in,
        kernel_initializer='he_normal',
        dropout_keep_prob=0.8, 
        kernel_size=17, 
        preactivation=True,
        postactivation_bn=False, 
        activation_function='relu'
    ):
        super(ResidualUnit, self).__init__()
        self.n_samples_out = n_samples_out
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        ##### layer ######
        self.conv1 = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=16, stride=1, padding='same', bias= False)
        # @TODO downsample = 4 always? 
        self.conv2 = nn.Conv1d(self.n_filters_out, self.n_filters_out, kernel_size=16, stride=4, padding=6, bias= False)
        self.drop1 = nn.Dropout(p=self.dropout_rate)
        self.bn1 = nn.BatchNorm1d(self.n_filters_out)   
        self.conv_skip = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=16, stride=1, padding='same', bias= False)
        self.activition = nn.ReLU(inplace=True) 
        # self.conv_skip = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1, padding=0, bias= False)
        
    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            # print (y.shape)
            maxpool1 = nn.MaxPool1d(downsample, stride=downsample)
            y = maxpool1(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            # print (y.shape)
            y = self.conv_skip(y)

        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = self.activition(x)
            x = self.bn1(x)
        else:
            x = self.bn1(x)
            x = self.activition(x)
        return x
    
    def forward(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[2]#y.shape[2]
        # downsample = n_samples_in // self.n_samples_out
        downsample = 4
        n_filters_in = y.shape[1]#y.shape[1]
        # print (n_samples_in, self.n_samples_out)

        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        #self.conv1 = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1, padding=0, bias= False)
        x = self.conv1(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = self.drop1(x)
        # print (x.shape, y.shape)
        # 2nd layer
        #self.conv2 = nn.Conv1d(self.n_filters_out, self.n_filters_out, kernel_size=1, stride=4, padding=0, bias= False)
        # print (x.shape, y.shape)
        x = self.conv2(x)
        # print (x.shape, y.shape)
        if self.preactivation:
            # print (x.shape, y.shape)
            x = torch.add(x,y)  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = self.drop1(x)
        else:
            x = self.bn1(x)
            x = torch.add(x,y)  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = self.drop1(x)
            y = x
        return [x, y]


class ECG_Survival_ResNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 5120

    ):
        super(ECG_Survival_ResNet, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1
        self.kernel_size = 16
        self.conv1 = nn.Conv1d(12, 64, kernel_size=16, stride=1, padding='same', bias= False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.kernel_initializer = 'he_normal'
        self.res1 = ResidualUnit(input_size//4, 128, 64, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res2 = ResidualUnit(input_size//16, 196, 128, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res3 = ResidualUnit(input_size//64, 256, 196, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res4 = ResidualUnit(input_size//256, 320, 256, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.flatten = nn.Flatten()
        self.dense_agsx = nn.Linear(2, 10)
        self.dense = nn.Linear(input_size//256*320, num_classes)

    def forward(self, x):
        
        signal = x[0]
        age_sex = x[1]
        
        x1 = self.conv1(signal)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1, y = self.res1([x1, x1])
        x1, y = self.res2([x1, y])
        x1, y = self.res3([x1, y])
        x1, _ = self.res4([x1, y])
        x1 = self.flatten(x1)

        # x2 = self.dense_agsx(age_sex)
        # print (x1.shape)
        x1 = self.dense(x1)
        # print (x1.shape)
        result = torch.cat([x1, age_sex], dim=1)
        return result
    
class ECG_ResNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 5120
    ):
        super(ECG_ResNet, self).__init__()
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1
        self.kernel_size = 16
        self.conv1 = nn.Conv1d(12, 64, kernel_size=1, stride=1, padding=0, bias= False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.kernel_initializer = 'he_normal'
        #4096
        self.res1 = ResidualUnit(input_size//4, 128, 64, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res2 = ResidualUnit(input_size//16, 196, 128, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res3 = ResidualUnit(input_size//64, 256, 196, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res4 = ResidualUnit(input_size//256, 320, 256, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.flatten = nn.Flatten()
        self.dense_agsx = nn.Linear(2, 10)
        self.dense = nn.Linear(input_size//256*320 + 10, num_classes)

    def forward(self, x):
        signal = x[0]
        age_sex = x[1]
        
        x1 = self.conv1(signal)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        # print (x1.shape)

        x1, y = self.res1([x1, x1])
        x1, y = self.res2([x1, y])
        x1, y = self.res3([x1, y])
        x1, _ = self.res4([x1, y])
        x1 = self.flatten(x1)

        x2 = self.dense_agsx(age_sex)
        x = torch.cat([x1, x2], dim=1)
        
        x = self.dense(x)
        result = self.sigmoid(x)
        # print (result.shape)        
        return result

    
# DATA LOADER
import importlib
import torch
from torch.utils.data import Dataset, DataLoader

import re
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath("../../src"))

import torch.nn.functional as f
from multiprocessing import Process
import gzip

import torchtuples as tt

class ECG_Dataset_Diagonal_labels(Dataset):
    def __init__(self, df = None, transform=None, age_sex_path_df = None, diago_df=None, y = None,idxs=None):
        self.ecg_df = df
        self.transform = transform
        self.diago_df = diago_df
        self.age_sex_path_df = age_sex_path_df
        self.y = y
        self.idxs = idxs
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        
        # row = self.ecg_df.iloc[idx]
        # time = self.ecg_df['time'].iloc[idx]
        # event = self.ecg_df['event'].iloc[idx]
        # base_name = row.datetime_ecgId
        base_name = self.idxs[item]
        
        # 1 get leads                        
        try: 
            np_path = "/home/padmalab/ecg/data/processed/ecgs_compressed/ecgs_np/%s.xml.npy.gz"%base_name
            f = gzip.GzipFile(np_path, "r")
            enp = np.load(f)[:,0:5120]
        except Exception as e:
            enp = np.zeros([12,5120])
            # try:
            #     ecg_leads = get_leads(row["path"])
            #     enp = np.ndarray([12,4096])
            #     for i,lead in enumerate(ecg_leads):
            #         enp[i] = lead["data"][0:4096]
            # except:
            #     # print('Exception', e)
            #     enp = np.zeros([12,4096])
            #     pass
            
        try:
            as_measures = self.age_sex_path_df.loc[base_name]
            as_measures = as_measures.to_numpy()
            # as_measures = as_measures.astype(np.float)
        except Exception as e:
            # print (e)
            as_measures = np.array([50, 1])
            
        y = self.ecg_df.loc[[base_name]].to_numpy()
        # agsx = self.age_sex_path_df.loc[[base_name]].to_numpy()
        sample = {
                "path": base_name,
                # "time": max(time,0),
                # "event": event,  
                "y": y.squeeze(),
                "leads": torch.FloatTensor(enp),
                "agsx": as_measures.astype(np.float32),
                }
        if self.transform:
            sample = self.transform(sample)
        # print (y.to_numpy())
#         return tt.tuplefy((sample['leads'],sample['agsx']), (sample['time'], sample['event']))
        return tt.tuplefy((sample['leads'], sample['agsx']), sample['y'])



class ClientUpdate(object):
    def __init__(self, dataset, agsx, batchSize, learning_rate, epochs, client, idxs, val_list):
        self.train_loader = DataLoader(ECG_Dataset_Diagonal_labels(df = dataset, age_sex_path_df = agsx, idxs = idxs), batch_size=batchSize, shuffle=True, num_workers=3)
        self.fl_data = dataset
        self.agsx = agsx
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.client = client
        self.val_list = val_list

    def train(self, model):

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=self.epochs,
            target_epsilon= 5,
            target_delta=1e-5,
            max_grad_norm=1.2,
        )
        
        early_stopping = EarlyStopping(patience=10, verbose=True, path =  "FL_DP5_models/fl_client%s_resnet_weights.pt"%self.client)
   

        e_loss = []
        for epoch in range(1, self.epochs+1):
            print("Epoch:", epoch)
            train_loss = 0.0

            model.train()
            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=128, 
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for data, labels in tqdm(memory_safe_data_loader):
                    if torch.cuda.is_available():
                        data, labels = [data[0].to(device),data[1].to(device)], labels.to(device)

                        # clear the gradients
                        optimizer.zero_grad()

                        output = model(data)
                        
                        loss = criterion(output.float(), labels.float())
                        loss.backward()
                        
                        optimizer.step()

                        train_loss += loss.item()*data[0].size(0)

            train_loss = train_loss/len(self.train_loader.dataset)
            e_loss.append(train_loss)
            
            val_loss = testing(model, self.fl_data, self.agsx, self.val_list, 256, criterion)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        total_loss = sum(e_loss)/len(e_loss)
        model.load_state_dict(torch.load("FL_DP5_models/fl_client%s_resnet_weights.pt"%self.client))
        return model.state_dict(), total_loss

    
def training(model, rounds, batch_size, lr, ds, agsx, train_data_dict, val_data_dict, S_t, E, plt_title, plt_color):
    """
    Function implements the Federated Averaging Algorithm from the FedAvg paper.
    Specifically, this function is used for the server side training and weight update

    Params:
    - model:           PyTorch model to train
    - rounds:          Number of communication rounds for the client update
    - batch_size:      Batch size for client update training
    - lr:              Learning rate used for client update training
    - ds:              Dataset used for training
    - data_dict:       Type of data partition used for training (IID or non-IID)
    - C:               Fraction of clients randomly chosen to perform computation on each round
    - K:               Total number of clients
    - E:               Number of training passes each client makes over its local dataset per round
    - tb_writer_name:  Directory name to save the tensorboard logs
    Returns:
    - model:           Trained model on the server
    """

    # global model weights
    global_weights = model.state_dict()
   
    # training loss
    train_loss = []

    # measure time
    # start = time.time()

    for curr_round in range(1, rounds+1):
        w, local_loss = [], []
        S_t = S_t
        
            
        # m = max(int(C*K), 1)
        
        # S_t = np.random.choice(range(K), m, replace=False)
        for k in S_t[curr_round-1]:
            print("Training on Hospital"+ str(k))
            local_update = ClientUpdate(dataset=ds, agsx = agsx, batchSize=batch_size, 
                                        learning_rate=lr, epochs=E, client= k, idxs=train_data_dict[k],val_list = val_data_dict[k])
            weights, loss = local_update.train(model=copy.deepcopy(model))

            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))

        # updating the global weights
        weights_avg = copy.deepcopy(w[0])
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]

            weights_avg[k] = torch.div(weights_avg[k], len(w))
            

        global_weights = weights_avg
        global_weights = {k.replace("_module.", ""): v for k, v in global_weights.items()}
        
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        print('Round: {}... \tAverage Loss: {}'.format(curr_round, round(loss_avg, 3)))
        train_loss.append(loss_avg)

#     end = time.time()
#     fig, ax = plt.subplots()
#     x_axis = np.arange(1, rounds+1)
#     y_axis = np.array(train_loss)
#     ax.plot(x_axis, y_axis, 'tab:'+plt_color)

#     ax.set(xlabel='Number of Rounds', ylabel='Train Loss',
#        title=plt_title)
#     ax.grid()
#     fig.savefig(plt_title+'.jpg', format='jpg')
#     print("Training Done!")
#     print("Total time taken to Train: {}".format(end-start))

    torch.save(model.state_dict(), 'FL_DP5_models/fl_resnet_localdp_0.5_weights.pt')
    return model


from torchmetrics import AUROC
from sklearn.metrics import roc_auc_score

def testing(model, dataset, agsx, data_list, bs, criterion):
    #test loss 
    test_loss = 0.0
    # correct_class = list(0. for i in range(num_classes))
    # total_class = list(0. for i in range(num_classes))

    test_loader = DataLoader(ECG_Dataset_Diagonal_labels(df = dataset, age_sex_path_df = agsx, idxs = data_list), batch_size = bs, num_workers=3)
    l = len(test_loader)
    # print(l)
    model.eval()
    preds = np.empty((0,10))
    true_lab = np.empty((0,10), int)
    for data, labels in tqdm(test_loader):
        # data, labels = data.to(device), labels.to(device)
        if torch.cuda.is_available():
            data, labels = [data[0].to(device),data[1].to(device)], labels.to(device)
        output = model(data)
        
        # print(labels.shape)
        true_lab = np.append(true_lab, np.array(labels.cpu().detach()), axis=0)
        # print(true_lab.shape)
    
        preds = np.append(preds, np.array(output.cpu().detach()), axis = 0)
        # print(preds)
        # print(preds.shape)
        # print(true_lab)
        # print(true_lab.shape)
        loss = criterion(output.float(), labels.float())
        test_loss += loss.item()*data[0].size(0)

#         _, pred = torch.max(output, 1)

#         correct_tensor = pred.eq(labels.data.view_as(pred))
#         correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
    
        #test accuracy for each object class
        # for i in range(len(labels)):
        #     label = labels.data[i]
        #     correct_class[label] += correct[i].item()
        #     total_class[label] += 1
    
      # avg test loss
    # print(preds)
    # print(true_lab.int())
    
    
    roc = roc_auc_score(true_lab.tolist(), preds.tolist())
    print("Val AUROC: {:.6f}\n".format(roc*100))
    test_loss = test_loss/len(test_loader.dataset)
    print("Val Loss: {:.6f}\n".format(test_loss))
    
    # pred_dict = {0: preds, 1: true_lab}
    # with open('/data/padmalab/ecg/data/processed/fl_dl_temp/fl_resnet_localdp_0.5_pred_dict.pickle', 'wb') as file:
    #     pickle.dump(pred_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    return test_loss

import pickle
import random

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

import warnings
warnings.simplefilter("ignore")


# number of training rounds
rounds = 3

h_t = [i for i in range(7)]
random.shuffle(h_t)
S_t = {}
S_t[0] = h_t[:2]
S_t[1] = h_t[2:4]
S_t[2] = h_t[4:]

for i in range(3):
    print(f"group {i+1}:",S_t[i])

    
E = 100
batch_size = 256
lr=0.01


with open('datasets_fl/full_hospitalwise_development_train.pickle', 'rb') as f:
    train_data_dict = pickle.load(f)

with open('datasets_fl/full_hospitalwise_development_val.pickle', 'rb') as f:
    val_data_dict = pickle.load(f)
    
    
    
# load model
resnet = ECG_ResNet()
resnet = ModuleValidator.fix(resnet)
ModuleValidator.validate(resnet, strict=False)
resnet = resnet.to(device)

# resnet = ECG_ResNet()
# if torch.cuda.is_available():
#     resnet = nn.DataParallel(resnet)
#     resnet.cuda()

resnet_trained = training(resnet, rounds, batch_size, lr, fl_data, agsx, train_data_dict, val_data_dict, S_t, E,  "FL_ResNet", "orange")


# resnet_trained = ECG_ResNet().to(device)
# resnet_trained.load_state_dict(torch.load("FL_DP5_models/fl_resnet_localdp_0.5_weights.pt"))

print('---------------Test metrics-------------') 

with open('datasets_fl/full_hospitalwise_holdout.pickle', 'rb') as f:
    test_data_dict = pickle.load(f)
    
test_data_list = []
for i in test_data_dict:
    test_data_list = test_data_list + test_data_dict[i]
    if i==7:
        break
    

criterion = nn.BCELoss()
result  = testing(resnet_trained, fl_data, agsx, test_data_list, 256, criterion)

for i in range(7):
    test_data_list = test_data_dict[i]
    criterion = nn.BCELoss()
    print("testing on site-",i+1)
    result  = testing(resnet_trained, fl_data, agsx, test_data_list, 256, criterion)