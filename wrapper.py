import csv
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils
from models import *


class Wrapper:
    def __init__(self, config, RESULT_DIR):
        utils.fix_seed(config.getint("HYPER_PARAMETERS", "seed"))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.set_dataset("five")
        self.set_model(model_name="deep")
        self.set_optimizer(optimizer_name=config.get("HYPER_PARAMETERS", "optimizer"))

        self.EPOCHS = 2000000

        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[self.EPOCHS * 0.3, self.EPOCHS * 0.6, self.EPOCHS * 0.8],
            gamma=0.2,
        )

        self.RESULT_DIR = RESULT_DIR

        self.SAVE_EPOCHS = [
            1,
            10,
            100,
            1000,
            10000,
            20000,
            30000,
            40000,
            50000,
            60000,
            70000,
            80000,
            90000,
            100000,
            1000000,
            2000000,
        ]

        self.LOG_PATH = os.path.join(self.RESULT_DIR, "log.txt")
        self.fieldnames = ["train loss", "test loss"]
        with open(self.LOG_PATH, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def set_dataset(self, image):
        if image == "five":
            train_images, _, test_images, _ = utils.get_images()
            X_train = train_images[0].unsqueeze(0).to(self.device)
            self.X_train = X_train
        elif image == "zero":
            train_images, _, test_images, _ = utils.get_images()
            X_train = train_images[1].unsqueeze(0).to(self.device)
            self.X_train = X_train
        elif image == "black":
            train_images = torch.tensor(np.zeros(28 * 28)).reshape(-1, 28 * 28)
            X_train = train_images.unsqueeze(0).to(self.device)

        ds_train = TensorDataset(X_train, X_train)

        X_test = test_images.to(self.device)
        ds_test = TensorDataset(X_test, X_test)

        self.loader_train = DataLoader(ds_train, batch_size=1, shuffle=True)
        self.loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

    def set_model(self, model_name):
        if model_name == "deep":
            self.model = deep_net().double().to(self.device)
        elif model_name == "shallow":
            self.model = shallow_net().double().to(self.device)

    def set_optimizer(self, optimizer_name):
        if optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        elif optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        elif optimizer_name == "Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=0.01)
        elif optimizer_name == "Adamax":
            self.optimizer = optim.Adamax(self.model.parameters(), lr=0.01)
        elif optimizer_name == "RMSprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.01)
        elif optimizer_name == "Rprop":
            self.optimizer = optim.Rprop(self.model.parameters(), lr=0.01)

    def train(self):
        self.model.train()
        for data, target in self.loader_train:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            lossval = loss.item()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        return lossval

    def test(self):
        self.model.eval()
        loss = 0
        mse = nn.MSELoss()

        for data, _ in self.loader_test:
            *_, output = self.model(data)
            loss += mse(
                self.X_train.reshape(-1, 28 * 28), output.reshape(-1, 28 * 28)
            ).item()

        data_num = len(self.loader_test.dataset)
        lossval = loss / data_num

        return lossval

    def execute(self):
        self.save(self.model, 0)

        for epoch in tqdm(range(1, self.EPOCHS + 1)):
            train_loss = self.train()

            if epoch % 100 == 0:
                test_loss = self.test()

                with open(self.LOG_PATH, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writerow({"train loss": train_loss, "test loss": test_loss})

            if epoch in self.SAVE_EPOCHS:
                self.save_model(self.model, epoch)

    def save_model(self, model, epoch):
        save_dir = os.path.join(self.RESULT_DIR, "model_" + str(epoch) + ".pth")
        torch.save(model.state_dict(), save_dir)
