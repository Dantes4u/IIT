import pandas as pd
import numpy as np
import os
import json

from experiment import VariablePlot, get_dirs, roc_plot, dist_plot, confusion_plot, mae_plot, accuracy_plot, pr_plot


class Experiment:
    def __init__(self, domains, config, experiment_dir):
        self.plots = {
            'ROC': roc_plot,
            'PR': pr_plot,
            'Distribution': dist_plot,
            'Confusion': confusion_plot,
            'MAE': mae_plot,
            'Accuracy': accuracy_plot,
        }
        self.domains = domains
        self.domain_names = {domain['name']: domain for domain in self.domains}

        self.config = config
        self.settings = self.config['settings']

        self.basedir = experiment_dir
        self.experiment_dirs = {
            'plot': os.path.join(self.basedir, 'Plots'),
            'dump': os.path.join(self.basedir, 'Dump')
        }

        self.splits = {'Overall': {'data': pd.DataFrame(dtype=np.int32)}}
        self.epoch = 0

        self.track_variables = {'Loss': [VariablePlot('Loss', self.settings)]}
        for domain in domains:
            domain_name = domain['name']
            self.track_variables[domain_name] = []
            if domain['ordinal']:
                var_names = self.config['ordinal']['vars']
            else:
                var_names = self.config['categorical']['vars']
            for var_name in var_names:
                variable = VariablePlot(var_name, self.settings, domain)
                self.track_variables[domain_name].append(variable)

    def setup(self, epoch):
        self.splits = {'Overall': {'data': pd.DataFrame(dtype=np.int32)}}
        self.epoch = epoch

    def update_split(self, split):
        split['data'] = split['data'].dropna(how='any')
        split_data = split['data'].copy()
        if split_data.empty:
            return

        for domain_name, domain in self.domain_names.items():
            domain = self.domain_names[domain_name]
            if domain['ordinal']:
                bin_field = '{}-bin'.format(domain_name)
                split_data.loc[:, bin_field] = ''
                min_value, max_value = domain['min_value'], domain['max_value']
                step = self.settings['step'][domain_name]
                bins = np.arange(min_value, max_value + 1, step)
                if bins[-1] != max_value:
                    bins = np.append(bins, max_value)
                bins[-1] += 1

                for i in range(1, bins.size):
                    condition = (bins[i - 1] <= split_data[domain_name]) & (split_data[domain_name] < bins[i])
                    split_data.loc[condition, bin_field] = '{}:{}'.format(bins[i - 1], bins[i])

        self.splits[split['name']] = {'data': split_data}

        if split['name'].lower() != 'train':
            test_data = self.splits['Overall']['data'].append(split_data, sort=True, ignore_index=True)
            self.splits['Overall']['data'] = test_data

    def render(self):
        experiment_dirs = self.experiment_dirs
        if self.epoch != -1:
            experiment_dirs = get_dirs(self.experiment_dirs, '{:03d}'.format(self.epoch))

        var_dirs = get_dirs(experiment_dirs, 'variables')
        variables = self.track_variables

        for split_name in self.splits:
            data = self.splits[split_name]['data']
            if data.empty:
                continue
            for domain_name in variables:
                for variable in variables[domain_name]:
                    variable.set(split_name=split_name, data=data, epoch=self.epoch)

        metrics = {}
        for domain_name in variables:
            for variable in variables[domain_name]:
                variable.render(var_dirs, metrics)

        dist_plots = self.epoch in (-1, 0)

        for split_name in self.splits:
            data = self.splits[split_name]['data']
            if data.empty:
                continue
            for domain in self.domains:
                if domain['ordinal']:
                    plot_names = self.config['ordinal']['plots']
                else:
                    plot_names = self.config['categorical']['plots']
                for plot_name in plot_names:
                    if plot_name == 'Distribution' and not dist_plots:
                        continue
                    plot_dirs = experiment_dirs
                    if split_name.lower() not in ['overall', 'train']:
                        plot_dirs = get_dirs(plot_dirs, os.path.join('splits', plot_name.lower()))
                    plot = self.plots[plot_name]
                    plot(data=data, split_name=split_name, domain=domain, settings=self.settings,
                         output_dirs=plot_dirs, metrics=metrics)

        if self.epoch == -1:
            output_path = os.path.join(self.basedir, 'Metrics.json')
        else:
            output_path = os.path.join(self.basedir, 'Metrics')
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, 'metrics-{:03d}.json'.format(self.epoch))

        with open(output_path, 'w+') as output_file:
            json.dump(metrics, output_file, indent=4)


