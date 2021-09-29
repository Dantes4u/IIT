import numpy as np
import json
import torch

class DataHolder:
    def __init__(self, config):
        self.train_pth = config['train_markup']
        self.test_pth = config['test_markup']
        self.mapping_train = {}
        self.mapping_test = {}
        self.data_pth = config['data_dir']

        with open(self.train_pth, "r") as train_file:
            train_file = json.load(train_file)
            self.train = train_file
        with open(self.test_pth, "r") as test_file:
            test_file = json.load(test_file)
            self.test = test_file

        self.domains = config['domains']
        for domain in self.domains:
            for name in ('ordinal', 'categorical'):
                domain[name] = domain.get(name, False)

        all_labels = []
        for i in range(len(self.train)):
            all_labels.append({"no fall":0, "fall":1}[self.train[str(i)]['label']])
        all_labels = np.array(all_labels)
        self.weights = []
        weights = np.array([1 / (2 * np.mean(all_labels == label)) for label in range(2)])
        self.weights.append(torch.from_numpy(weights.astype(np.float32)))
        print(self.weights[0])
        self.weights[0][1] = 0.5
    def get_dataset(self, train = True):
        if train == True:
            data = self.train
            name = 'Train'
        else:
            data = self.test
            name = 'Test'
        dataset = {
            'data_pth': self.data_pth,
            'name': name,
            'data': data,
            'size': len(data),
            'train': name == 'Train'
        }
        return dataset