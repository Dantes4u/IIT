import os
import numpy as np
import yaml
import random
import json
import torch
import torch.nn as nn
from tqdm import tqdm

import models

from logs import Logger
from experiment import Experiment, SplitExperiment
from data import DataHolder, DataIter
from ranger21 import Ranger21
from torch.nn.parallel import DistributedDataParallel as DDP
from loss import FocalLoss, LabelSmoothingLoss, LabelSmoothingCrossEntropy

class Trainer:
    def __init__(self, logs_dir, config_name, gpu):
        self.gpu = gpu
        self.config = self._load_config(config_name)
        self.logs_dir = logs_dir
        if self.gpu == 0:
            self.logger = Logger()

        self.params = self.config['Parameters']
        self.data_config = self.config['Data']
        self.model_config = self.config['Model']
        self.transform_config = self.config['Transform']
        self.experiment_config = self.config['Experiment']

        self.distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
        if self.distributed:
            torch.cuda.set_device(gpu)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.num_gpus = torch.distributed.get_world_size()
        else:
            self.num_gpus = 1

        self.data_holder = DataHolder(self.data_config)
        self.num_epochs = self.params['num_epochs']
        self.batch_size = self.params['batch_size']

        self._seed()
        self._init_model()
        self._init_loaders()
        self._init_optimizer()
        self._init_scheduler()
        self._init_loss()
        self._init_experiment()

    def _seed(self):
        seed_state = {}
        for name in ('random', 'torch', 'numpy'):
            seed_state[name] = np.random.randint(0, 900)
            if self.distributed:
                seed_state[name] += torch.distributed.get_rank()

        random.seed(seed_state['random'])
        torch.manual_seed(seed_state['torch'])
        torch.cuda.manual_seed(seed_state['torch'])
        np.random.seed(seed=seed_state['numpy'])

        if self.gpu == 0:
            with open(os.path.join(self.logs_dir, 'seed_state.json'), 'w+') as output_file:
                json.dump(seed_state, output_file)
            
    def _init_model(self):
        output_sizes = []
        for domain in self.data_holder.domains:
            size = domain['length']
            if size == 2:
                size = 1
            output_sizes.append(size)

        model_name = self.model_config['name']
        self.model = models.MODELS[model_name](config=self.model_config)

        if 'params' in self.model_config:
            self.model.load_state_dict(torch.load(self.model_config['params']))

        self.model = self.model.cuda()
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu], output_device=self.gpu)
        # self.model = nn.DataParallel(self.model).cuda()

    def _init_scheduler(self):
        train_iters = len(self.data_holder.get_dataset(train=True)['data']) // self.batch_size + 1
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            train_iters * self.num_epochs,
        )

    def _init_optimizer(self):
        initial_lr = float(self.params['initial_lr'])
        wd = float(self.params['weight_decay'])
        lookahead_mergetime = int(self.params['lookahead_mergetime'])
        self.optimizer = Ranger21(
            self.model.parameters(),
            lookahead_mergetime=lookahead_mergetime,
            lr=initial_lr,
            weight_decay=wd,
            num_epochs=self.num_epochs,
            num_batches_per_epoch=self.train_iters
        )
        self.scaler = torch.cuda.amp.GradScaler()

    def _init_loss(self):
        """self.losses = [nn.BCEWithLogitsLoss if domain['length'] == 2 else nn.CrossEntropyLoss
                       for domain in self.data_holder.domains]
        loss_kwargs = [
            {'ignore_index': -1, 'weight': weights} if domain['length'] != 2 else {'pos_weight': weights[1]}
            for domain, weights in zip(self.data_holder.domains, self.data_holder.weights)
        ]
        self.losses = [loss(reduction='none', **kwargs).cuda() for loss, kwargs in zip(self.losses, loss_kwargs)]"""

        self.losses = [FocalLoss if domain['length'] == 2 else nn.CrossEntropyLoss#LabelSmoothingCrossEntropy#LabelSmoothingLoss
                       for domain in self.data_holder.domains]
        loss_kwargs = [
            {'weight': weights} if domain['length'] != 2 else {'weight': weights[1]/weights[0]}
            for domain, weights in zip(self.data_holder.domains, self.data_holder.weights)
        ]
        self.losses = [loss(**kwargs).cuda() for loss, kwargs in zip(self.losses, loss_kwargs)]

    def _init_loaders(self):
        if self.gpu == 0:
            self.logger.info("Building random loader for train data")
        train_iter = DataIter(self.data_holder, gpu=self.gpu, train=True, config = self.transform_config)
        self.sampler = torch.utils.data.distributed.DistributedSampler(train_iter) if self.distributed else None
        self.train_loader = torch.utils.data.DataLoader(
            train_iter,
            sampler=self.sampler,
            batch_size=self.batch_size,
            num_workers=self.params['num_workers'],
            shuffle=(self.sampler is None),
            pin_memory=self.params['pin_memory'],
            prefetch_factor=self.params['prefetch_factor']
        )
        self.train_iters = len(self.train_loader)

        if self.gpu == 0:
            self.logger.info("Building loader for test data")
        test_iter = DataIter(self.data_holder, gpu=0, train=False, config =self.transform_config)
        self.test_loader = torch.utils.data.DataLoader(
            test_iter,
            batch_size=self.batch_size,
            num_workers=self.params['num_workers'],
            shuffle=False,
            pin_memory=self.params['pin_memory'],
            prefetch_factor=self.params['prefetch_factor']
        )
        self.test_iters = len(self.test_loader)
    def _init_experiment(self):
        self.experiment = Experiment(self.data_holder.domains, self.experiment_config, self.logs_dir)

    def train(self, epoch):
        if self.distributed:
            self.sampler.set_epoch(epoch)
        self.model.train()
        if self.gpu == 0:
            self.logger.info('Training epoch {}/{}.'.format(epoch, self.num_epochs))
            self.logger.info('Learning rate schedule: {}'.format(self.lr_scheduler.get_last_lr()[0]))
            self.experiment.setup(epoch)
            split_experiment = SplitExperiment(domains=self.data_holder.domains, split_name='Train')
        iterator = self.train_loader
        if self.gpu == 0:
            iterator = tqdm(iterator, total=self.train_iters, unit='batch')
        for images, labels in iterator:
            images = images.cuda(self.gpu)
            labels = labels.cuda(self.gpu)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                prediction = self.model(images)
                # print(prediction.squeeze(dim=1))
                # print(labels.unsqueeze(dim=1).float())
                losses = self.losses[0](prediction, labels.unsqueeze(dim=1).float())
            loss = torch.mean(sum(losses))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            if self.gpu == 0:
                losses = [loss.cpu().detach().numpy().astype(np.float32)]
                labels = [labels.cpu().detach().numpy()]
                predictions = [prediction.cpu().detach().float()]
                split_experiment.update(predictions, labels, losses)
                iterator.set_description('loss: {:.3f}'.format(np.mean(loss.cpu().detach().numpy())))

        if self.gpu == 0:
            self.experiment.update_split(split_experiment.get_split())

    def test(self, epoch):
        self.model.eval()
        self.logger.info('Validating epoch {} at \'{}\'.'.format(epoch, "test"))
        loader = self.test_loader
        iters = self.test_iters
        split_experiment = SplitExperiment(domains=self.data_holder.domains, split_name="test")
        iterator = tqdm(loader, total=iters, unit='batch')
        for images, labels in iterator:
            images = images.cuda(self.gpu)
            labels = labels.cuda(self.gpu)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    prediction = self.model(images)
                    losses = self.losses[0](prediction.squeeze(dim=1), labels.float())
                loss = torch.mean(sum(losses))
                losses = [loss.cpu().detach().numpy().astype(np.float32)]
                labels = [labels.cpu().detach().numpy()]
                predictions = [prediction.cpu().detach().float()]

                split_experiment.update(predictions, labels, losses)
                iterator.set_description('loss: {:.3f}'.format(np.mean(loss.cpu().detach().numpy())))
        self.experiment.update_split(split_experiment.get_split())

    def render(self):
        self.experiment.render()

    def export_model(self, epoch):
        model_name = self.model_config['name'].lower()
        model_output_dir = os.path.join(self.logs_dir, 'Models', '{:03d}'.format(epoch))
        os.makedirs(model_output_dir, exist_ok=True)
        model_output_path = os.path.join(model_output_dir, f'{model_name}_{epoch}.pt')
        state_dict = self.model.module.state_dict() if self.distributed else self.model.state_dict()
        torch.save(state_dict, model_output_path)

    @staticmethod
    def _load_config(config_name):
        config_path = os.path.join('config', 'base', config_name)
        with open(config_path, 'r') as input_file:
            config = yaml.safe_load(input_file)
        return config


def train_model(logs_dir, config_name, gpu):
    trainer = Trainer(logs_dir, config_name, gpu)
    for epoch in range(trainer.num_epochs):
        trainer.train(epoch)
        if gpu == 0:
            trainer.test(epoch)
            trainer.render()
            trainer.export_model(epoch)
