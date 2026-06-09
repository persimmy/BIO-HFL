#!/usr/bin/env python

# -*- coding: utf-8 -*-

# Python version: 3.6

import torch

from torch import nn, autograd

from torch.utils.data import DataLoader, Dataset

import numpy as np

import random

from sklearn import metrics

import sys

import os

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):

        self.dataset = dataset

        self.idxs = list(idxs)

    def __len__(self):

        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[self.idxs[item]]

        return image, label

class LocalUpdate(object):

    def __init__(self, args, dataset, idxs, local_ep=1, loader=None):

        self.args = args

        self.local_ep = local_ep

        self.loss_func = torch.nn.CrossEntropyLoss()

        if loader is not None:

            self.ldr_train = loader

        else:

            from torch.utils.data import DataLoader

            self.ldr_train = DataLoader(

                DatasetSplit(dataset, idxs),

                batch_size=self.args.local_bs,

                shuffle=True,

                num_workers=min(getattr(self.args, "num_workers", 0), 2),

                pin_memory=True,

                persistent_workers=True if getattr(self.args, "num_workers", 0) > 0 else False,

                prefetch_factor=2 if getattr(self.args, "num_workers", 0) > 0 else None

            )

    def train(self, net):

        net.train()

        # optimizer

        local_lr = float(getattr(self.args, "current_lr", self.args.lr))

        if self.args.optimizer == "SGD":

            optimizer = torch.optim.SGD(

                net.parameters(), lr=local_lr,

                momentum=self.args.momentum, weight_decay=self.args.weight_decay

            )

        elif self.args.optimizer == "Adam":

            optimizer = torch.optim.Adam(

                net.parameters(), lr=local_lr,

                weight_decay=self.args.weight_decay, amsgrad=True

            )

        else:

            raise ValueError("Invalid optimizer")

        epoch_loss = []

        n_samples_seen = 0

        for ep in range(self.local_ep):

            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                images = images.to(self.args.device, non_blocking=True)

                labels = labels.to(self.args.device, non_blocking=True)

                if images.size(0) == 1:

                    if self.args.verbose:

                        print(f"Skip singleton batch at local epoch {ep}, batch {batch_idx}")

                    continue
                optimizer.zero_grad(set_to_none=True)

                logits = net(images)

                loss = self.loss_func(logits, labels)

                loss.backward()

                optimizer.step()

                n_samples_seen += images.size(0)

                batch_loss.append(loss.item())

                if self.args.verbose and batch_idx % 10 == 0:

                    print(

                        'Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                            ep,

                            batch_idx * len(images),

                            len(self.ldr_train.dataset),

                            100. * batch_idx / max(1, len(self.ldr_train)),

                            loss.item()

                        )

                    )

                if self.args.verbose and (batch_idx + 1) % self.args.train_acc_batches == 0:

                    try:

                        m = net.module if hasattr(net, "module") else net

                        tsteps = getattr(m, "timesteps", None)

                        print(f'Epoch: {ep}, batch {batch_idx + 1}, leak {leak}, timesteps {tsteps}')

                    except Exception:

                        pass

            epoch_loss.append(sum(batch_loss) / max(1, len(batch_loss)))

        return net.state_dict(), (sum(epoch_loss) / max(1, len(epoch_loss))), n_samples_seen

