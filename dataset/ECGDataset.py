import numpy as np
from torch.utils.data import Dataset
import torch

class ECGDatasetEMD2DNPY(Dataset):
    def __init__(self, datas_idx, labels, path='/data/WangJilin/data/MIT-BIH/data_used', method='gram-npy', emd_num=4):
        super(ECGDatasetEMD2DNPY,self)
        self.idx = datas_idx
        emd_fold_path = f'{path}/emd-{emd_num}/'
        gasf_fold_path = f'{path}/{method}/'
        self.npy_data_paths = np.char.add(emd_fold_path,np.char.add(datas_idx.astype(str),'.npy'))
        self.gasf_data_paths = np.char.add(gasf_fold_path,np.char.add(datas_idx.astype(str),'.npy'))
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        # print(item)

        if torch.is_tensor(item):
            item = item.tolist()
        idx = torch.tensor(self.idx[item], dtype=torch.int)

        gasf_paths = self.gasf_data_paths[item]
        gasf = np.load(gasf_paths)
        gasf = torch.tensor(gasf,dtype=torch.float)

        emd_path = self.npy_data_paths[item]
        emd = np.load(emd_path)
        # print('0: ',idx)
        # print('1:', npy.shape)
        # print('2: ', npy.shape)
        emd = torch.tensor(emd,dtype=torch.float)

        label = torch.tensor(self.labels[item], dtype=torch.int)

        return idx, emd, gasf.unsqueeze(0), label



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # data = np.load('/data/WangJilin/data/MIT-BIH/data_used/data_label_5class.npy')
    data = np.load('/data/WangJilin/data/MIT-BIH/data_used/train_data_label_balance_5class.npy')
    data_idx = data[:,0]
    data_label = label = data[:,1]
    print(data_idx.shape,data_label.shape)

    sleepSet = ECGDatasetEMD2DNPY(datas_idx=data_idx,labels=data_label,path='/data/WangJilin/data/MIT-BIH/data_used/',method='gram-npy')
    dataloader = DataLoader(sleepSet, batch_size=4, shuffle=True)
    # featureSet = ECGFeatureSet(datas_idx=data_idx,labels=data_label)
    # dataloader = DataLoader(featureSet, batch_size=512, shuffle=True)
    for idx, npy, images, labels in dataloader:
        # Your training code here
        print(idx, npy.shape, images.shape,labels)

        break