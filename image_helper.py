from collections import defaultdict
import copy

import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet import ResNet18
from utils.utils import SubsetSampler

logger = logging.getLogger("logger")

class ImageHelper(Helper):

    def poison(self):
        return

    def create_model(self):
        logger.info('==> Init Local/Global Model...')
        if self.params['dataset'] == 'CIFAR-10':
            local_models = {pos: ResNet18(name='Local', created_time=self.params['current_time'], dataset='CIFAR-10', num_classes=10) for pos in range(self.params['number_of_total_participants'])}
            target_model = ResNet18(name='Target', created_time=self.params['current_time'], dataset='CIFAR-10', num_classes=10)
            target_model = target_model.cuda()

        elif self.params['dataset'] == 'CIFAR-100':
            local_models = {pos: ResNet18(name='Local', created_time=self.params['current_time'], dataset='CIFAR-100', num_classes=100) for pos in range(self.params['number_of_total_participants'])}
            target_model = ResNet18(name='Target', created_time=self.params['current_time'], dataset='CIFAR-100', num_classes=100)
            target_model = target_model.cuda()

        elif self.params['dataset'] == 'FMNIST':
            local_models = {pos: ResNet18(name='Local', created_time=self.params['current_time'], dataset='FMNIST', num_classes=10) for pos in range(self.params['number_of_total_participants'])}
            target_model = ResNet18(name='Target', created_time=self.params['current_time'], dataset='FMNIST', num_classes=10)
            target_model.load_state_dict(torch.load('./global_models/fmnist_target_model.pth'))
            target_model = target_model.cuda()

        elif self.params['dataset'] == 'EMNIST':
            local_models = {pos: ResNet18(name='Local', created_time=self.params['current_time'], dataset='EMNIST', num_classes=10) for pos in range(self.params['number_of_total_participants'])}
            target_model = ResNet18(name='Target', created_time=self.params['current_time'], dataset='EMNIST', num_classes=10)
            target_model = target_model.cuda()


        if self.params['resumed_model']:
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_models = local_models
        self.target_model = target_model


    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """ Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def poison_dataset(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        indices = list()

        # create array that starts with poisoned images
        # create candidates:
        range_no_id = list(range(len(self.train_dataset.data)))
        if self.params['modify_poison']:
            for image in self.params['poison_images'] + self.params['poison_images_test']:
                if image in range_no_id:
                    range_no_id.remove(image)

            # add random images to other parts of the batch
            for batches in range(0, self.params['size_of_secret_dataset']):
                range_iter = random.sample(range_no_id, self.params['batch_size'])
                indices.extend(range_iter)

        else:
            indices = copy.deepcopy(range_no_id)
            for i in range_no_id:
                if i in cifar_classes[self.params['poison_label_swap']]:
                    indices.remove(i)

        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def poison_test_dataset(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(range(1000)))


    def load_data(self):
        logger.info('==> Loading Data...')

        ### data load
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_emnist = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])

        transform_fmnist = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        if self.params['dataset'] == 'CIFAR-10':
            self.train_dataset = datasets.CIFAR10('./data/CIFAR-10/rawdata',
                                                  train=True, download=True,
                                                  transform=transform_train) # Training dataset

            self.test_dataset = datasets.CIFAR10('./data/CIFAR-10/rawdata',
                                                 train=False,
                                                 transform=transform_test) # Test dataset

        elif self.params['dataset'] == 'CIFAR-100':
            self.train_dataset = datasets.CIFAR100('./data/CIFAR-100/rawdata',
                                                  train=True, download=True,
                                                  transform=transform_train) # Training dataset

            self.test_dataset = datasets.CIFAR100('./data/CIFAR-100/rawdata',
                                                 train=False,
                                                 transform=transform_test) # Test dataset

        elif self.params['dataset'] == 'EMNIST':
            self.train_dataset = datasets.EMNIST('./data/EMNIST/rawdata',
                                                 train=True, split='mnist',
                                                 download=True,
                                                 transform=transform_emnist)
            self.test_dataset = datasets.EMNIST('./data/EMNIST/rawdata',
                                                train=False, split='mnist',
                                                transform=transform_emnist)

        # set local client train/test dataset
        p_train_loaders = {}
        p_eval_loaders = {}
        for i in range(self.params['number_of_total_participants']):
            ## load the local client's training datasets
            with open('./data/{}/train/{}.npz'.format(self.params['dataset'], i), 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            X_train = torch.Tensor(train_data['x']).type(torch.float32)
            y_train = torch.Tensor(train_data['y']).type(torch.int64)
            p_train = [(X, y) for X, y in zip(X_train, y_train)]
            p_train_loader = torch.utils.data.DataLoader(p_train,
                                                         batch_size=self.params['batch_size'],
                                                         drop_last=False)
            p_train_loaders[i] = p_train_loader

            ## load the local client's training datasets
            with open('./data/{}/test/{}.npz'.format(self.params['dataset'], i), 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()

            X_eval = torch.Tensor(train_data['x']).type(torch.float32)
            y_eval = torch.Tensor(train_data['y']).type(torch.int64)
            p_eval = [(X, y) for X, y in zip(X_eval, y_eval)]
            p_eval_loader = torch.utils.data.DataLoader(p_eval,
                                                        batch_size=self.params['batch_size'],
                                                        drop_last=False)
            p_eval_loaders[i] = p_eval_loader

        self.train_data = p_train_loaders
        self.eval_data = p_eval_loaders

        # self.test_data = self.get_test()
        self.poisoned_data_for_train = self.poison_dataset()
        self.test_data_poison = self.poison_test_dataset()

    def get_secret_loader(self):
        """For poisoning we can use a larger data set. I don't sample randomly, though.
        """
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)
        shuffled_indices = indices[:self.params['size_of_secret_dataset']]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=SubsetSampler(shuffled_indices))

        return train_loader

    def get_test(self):
        """Test dataset"""
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)

        return test_loader

    def get_batch(self, bptt, evaluation=False):
        data, target = bptt
        data = data.cuda()
        target = target.cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target