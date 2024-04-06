import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from typing import Type, Any, Callable, Union, List, Optional


# from __future__ import print_function, division

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# from earlystopping import EarlyStopping
cudnn.benchmark = True

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import warnings
warnings.simplefilter("ignore")



def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        # nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
        
class ResidualUnit(nn.Module):

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
        self.conv1 = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1, padding=0, bias= False)
        # @TODO downsample = 4 always? 
        self.conv2 = nn.Conv1d(self.n_filters_out, self.n_filters_out, kernel_size=1, stride=4, padding=0, bias= False)
        self.drop1 = nn.Dropout(p=self.dropout_rate)
        self.bn1 = nn.BatchNorm1d(self.n_filters_out)   
        self.conv_skip = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1, padding=0, bias= False)
        
        self.activition = nn.ReLU(inplace=True) 
        
    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
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
        n_samples_in = y.shape[2]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[1]
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = self.conv1(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = self.drop1(x)

        # 2nd layer
        x = self.conv2(x)
        if self.preactivation:
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

class ECG_ResNet(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
    ):
        super(ECG_ResNet, self).__init__()
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1
        self.kernel_size = 16
        self.conv1 = nn.Conv1d(12, 64, kernel_size=1, stride=1, padding=0, bias= False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0, bias= False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.kernel_initializer = 'he_normal'
        #4096
        self.res1 = ResidualUnit(1024, 128, 64, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res2 = ResidualUnit(256, 196, 128, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res3 = ResidualUnit(64, 256, 196, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.res4 = ResidualUnit(16, 320, 256, kernel_size=self.kernel_size,
                        kernel_initializer=self.kernel_initializer)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool1d(905, stride=1)
        self.dense_agsx = nn.Linear(2, 10)
        self.dense = nn.Linear(5130, num_classes)

    def forward(self, signal):
        x1 = self.conv1(signal[0])
        # print("1st", x1.shape)
        x1 = self.bn1(x1)
        # print("2nd", x1.shape)
        x1 = self.relu(x1)
        # print("3rd", x1.shape)
        x1 = self.conv2(x1)
        # print("11st", x1.shape)
        x1 = self.bn1(x1)
        # print("22nd", x1.shape)
        x1 = self.relu(x1)
        # print("33rd", x1.shape)
        x1 = self.maxpool(x1)
        # print("pool", x1.shape)
        x1, y = self.res1([x1, x1])
        # print("4th", x1.shape, y.shape)
        x1, y = self.res2([x1, y])
        # print("5th", x1.shape, y.shape)
        x1, y = self.res3([x1, y])
        # print("6th", x1.shape, y.shape)
        x1, _ = self.res4([x1, y])
        # print("7th", x1.shape)
        x1 = self.flatten(x1)
        # print("8th", x1.shape)
        x2 = self.dense_agsx(signal[1])
        # print(x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print(x.shape)
        x = self.dense(x)
        result = self.sigmoid(x)
        # print (result.shape)        
        return result
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import importlib
import torch
from torch.utils.data import Dataset, DataLoader

import re
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath("../../src"))

# from PySierraECG import get_leads
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
            enp = np.load(f)[:,0:5000]
        except Exception as e:
            enp = np.zeros([12,5000])
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

import pickle
fl_data = pd.read_pickle('fl_data_bin_10.pickle')
agsx = pd.read_pickle('agsx_df.pickle')
with open('final_nsplit_train.pickle', 'rb') as f:
    train_data_dict = pickle.load(f)
    
train_list = []
for i in train_data_dict:
    if i==7:
        break
    train_list += train_data_dict[i]
# train_list = train_data_dict[6]
# len(train_list)

with open('final_nsplit_test.pickle', 'rb') as f:
    test_dict = pickle.load(f)
test_data_list = []
for i in test_dict:
    if i==7:
        break
    test_data_list += test_dict[i]
# test_data_list = test_dict[6]
train_loader = DataLoader(ECG_Dataset_Diagonal_labels(df = fl_data, age_sex_path_df = agsx, idxs = train_list), batch_size=256,shuffle=True, num_workers=3)

def train_model(model, criterion, optimizer, num_epochs=25, phase='train', test_list = test_data_list, train_list = train_list):
    since = time.time()
    train_eloss_graph = []
    test_eloss_graph = []
    train_eauroc_graph = []
    test_eauroc_graph = []

    for epoch in range(num_epochs):
        
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        if phase=="train":
            model.train() 
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            # i=0
            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=128, 
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for inputs, labels in memory_safe_data_loader:
                    # print(i)
                    # i+=1
                    # print(inputs[0].shape, inputs[1].shape)
                    inputs = [inputs[0].to(device),inputs[1].to(device)]
                    # print(inputs)
                    # print(inputs.shape)
                    labels = labels.to(device)
                    # print(labels.shape)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # print(outputs.shape)
                        # print(outputs)
                        # print(labels.shape)
                        # print(labels)
                        # _, preds = torch.max(outputs, 1)
                        # _, true = torch.max(labels, 1)
                        # print(preds.shape)
                        # print(true.shape)
                        # print(preds)
                        loss = criterion(outputs.float(), labels.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs[0].size(0)
                # running_corrects += torch.sum(preds == true)
            # if phase == 'train':
            #     scheduler.step()
            print('Train metrics')
            train_result, train_auroc = testing(model, fl_data, agsx, train_list, 256, criterion)
            print('Test metrics')
            test_result, test_auroc = testing(model, fl_data, agsx, test_list, 256, criterion)

            epoch_loss = running_loss / len(train_loader.dataset)
            # epoch_acc = running_corrects.double() /1171
            train_eloss_graph.append(train_result)
            train_eauroc_graph.append(train_auroc)
            test_eloss_graph.append(test_result)
            test_eauroc_graph.append(test_auroc)
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
           
                # test_data_list = test_data_list[:256]
            
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    torch.save(model.state_dict(), '/data/padmalab/ecg/data/processed/fl_dl_temp/dp_0.5_resnet_plots_model.pt')
    epoch_count = range(1, len(train_eloss_graph) + 1)
    # Visualize loss history
    plt.plot(epoch_count, train_eloss_graph, 'r--')
    plt.plot(epoch_count, test_eloss_graph, 'b--')
    plt.legend(['Training Loss','Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('dp_0.5_resnet_plots_loss.jpg')
    plt.clf()
    plt.plot(epoch_count, train_eauroc_graph, 'r--')
    plt.plot(epoch_count, test_eauroc_graph, 'b--')
    plt.legend(['Training AUROC', 'Test AUROC'])
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.savefig('dp_0.5_resnet_plots_auroc.jpg')
    # print(eloss_graph)
    return model



from torchmetrics import AUROC
from sklearn.metrics import roc_auc_score
def testing(model, dataset, agsx, data_list, bs, criterion):
    #test loss 
    test_loss = 0.0
    # correct_class = list(0. for i in range(num_classes))
    # total_class = list(0. for i in range(num_classes))

    test_loader = DataLoader(ECG_Dataset_Diagonal_labels(df = dataset, age_sex_path_df = agsx, idxs = data_list), batch_size=bs, num_workers=3)
    l = len(test_loader)
    # print(l)
    model.eval()
    preds = np.empty((0,10))
    true_lab = np.empty((0,10), int)
    for data, labels in test_loader:
        # data, labels = data.to(device), labels.to(device)
        if torch.cuda.is_available():
            data, labels = [data[0].to(device), data[1].to(device)], labels.to(device)
        output = model(data)
        true_lab = np.append(true_lab, np.array(labels.cpu().detach()), axis=0)
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
    print(" AUROC: {:.6f}\n".format(roc*100))
    test_loss = test_loss/len(test_loader.dataset)
    print(" Loss: {:.6f}\n".format(test_loss))
    
    pred_dict = {0: preds, 1: true_lab}
    with open('/data/padmalab/ecg/data/processed/fl_dl_temp/dp_0.5_resnet_plots.pickle', 'wb') as file:
        pickle.dump(pred_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    return test_loss, roc*100



model = ECG_ResNet()
resnet = ModuleValidator.fix(model)
ModuleValidator.validate(resnet, strict=False)
model_ft = resnet.to(device)

criterion = nn.BCELoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
epochs = 20
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model_ft,
    optimizer=optimizer_ft,
    data_loader=train_loader,
    epochs=epochs,
    target_epsilon=0.5,
    target_delta=1e-5,
    max_grad_norm=1.2,
    # poisson_sampling = False
)
print(f"Using sigma={optimizer.noise_multiplier}")
# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# test_data_list = test_data_list[:256]

model_ft = train_model(model, criterion, optimizer,
                       num_epochs=epochs, test_list = test_data_list, train_list = train_list)

# model.load_state_dict(torch.load('/data/padmalab/ecg/data/processed/fl_dl_temp/site1_model.pt'))
# model_ft = model
        
# print('---------------Test metrics-------------')        
# criterion = nn.BCELoss()
# result, auroc = testing(model_ft, fl_data, agsx, test_data_list, 256, criterion)