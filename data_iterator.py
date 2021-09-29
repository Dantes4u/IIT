import numpy as np
import cupy as cp
import random
from PIL import Image
import torch
from data import TestTransform
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

class DataIter:
    def __init__(self, data_holder, gpu, train, config):
        super().__init__()
        self.gpu = gpu
        self.data_holder = data_holder
        self.train = train
        self.config = config
        self.dataset = self.data_holder.get_dataset(self.train)
        self.transform = TestTransform(self.config)
        self.init_worker = True
    def __getitem__(self, item):
        if self.init_worker:
            self.init_worker = False
            cp.cuda.Device(int(self.gpu)).use()
            np.random.seed(random.randint(0, 2**32))
            cp.random.seed(random.randint(0, 2**32))
        #print(self.dataset)
        elem = self.dataset['data'][str(item % self.dataset['size'])]
        label = {"no fall":0, "fall":1}[elem['label']]
        count1 = 0
        for i in elem['photoes']:
            splited = i.split("/")
            img_cur = np.load(f"{self.dataset['data_pth']}/{splited[-4]}~{splited[-2]}~{splited[-1][:-4]}.npy")
            img_cur = self.transform(image=img_cur)
            if count1 == 0:
                image = img_cur
            else:
                image = np.concatenate((image, img_cur), axis=2)
            count1+=1
        image = torch.as_tensor(image.transpose((2, 0, 1)), device=self.gpu)
        return image, label
    def __len__(self):
        return self.dataset['size']