import pandas as pd
import numpy as np
import os
import torch


class SplitExperiment:
    def __init__(self, domains, split_name):
        self.domains = domains
        self.split_name = split_name.capitalize()
        self.split_data = pd.DataFrame(dtype=np.int32)

    def update(self, predictions, labels, loss):
        data = {}
        for idx, domain in enumerate(self.domains):
            targets = labels[idx]
            pred = predictions[idx]
            domain_name = domain['name']
            if domain['categorical']:
                if pred.shape[1] == 1:
                    pred = np.squeeze(torch.sigmoid(pred).numpy(), axis=-1)
                    pred_value = pred > 0.5
                    data[domain_name + '-scores-0'] = 1 - pred
                    data[domain_name + '-scores-1'] = pred
                else:
                    pred = torch.softmax(pred, dim=1).numpy()
                    pred_value = np.argmax(pred, axis=1)
                    for i in range(pred.shape[1]):
                        data[domain_name + '-scores-{}'.format(i)] = pred[:, i]
            else:
                pred_value = np.rint(pred.numpy()).astype(np.int32, copy=False)

            data[domain_name] = targets
            data[domain_name + '-pred'] = np.rint(pred_value).astype(np.int32, copy=False)

            if loss is not None:
                data[domain_name + '-loss'] = loss[idx]
            else:
                data[domain_name + '-loss'] = -1

        if loss is not None:
            data['Loss'] = np.sum(np.array(loss), axis=0)
        else:
            data['Loss'] = -1

        self.split_data = self.split_data.append(pd.DataFrame(data), sort=True, ignore_index=True)

    def get_split(self):
        split = {
            'name': self.split_name,
            'data': self.split_data,
        }
        return split

    def save(self, output_dir):
        split_data = pd.DataFrame(dtype=np.int32)
        for domain in self.domains:
            domain_name = domain['name']
            pred_name = domain_name + '-pred'
            for name in (domain_name, pred_name):
                split_data[name] = self.split_data[name].copy()
                if not domain['ordinal']:
                    split_data[name] = split_data[name].replace(dict(enumerate(domain['values'])))
        path = os.path.join(output_dir, self.split_name + '.csv')
        split_data.to_csv(path, index=False)



