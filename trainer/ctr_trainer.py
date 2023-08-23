import sys

sys.path.append("..")

import os
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from basic.callback import EarlyStopper


class CTRTrainer(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=50,
        earlystop_patience=10,
        device="gpu",  # 这里改成了gpu，原本是cpu
        gpus=[0,1,2],  # 添加gpu
        model_path="./",
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if optimizer_params is None:  # 优化器参数
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.scheduler = None
        if scheduler_fn is not None:  # 学习率参数
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()  # default loss cross_entropy 将二元交叉熵损失设置为默认损失函数
        self.evaluate_fn = roc_auc_score  # default evaluate function 将ROC AUC得分函数设置为默认评估函数
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)  # 创建一个早停对象，基于提供的earlystop_patience参数
        self.model_path = model_path # 设置模型路径属性

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)  # 初始化一个进度条，跟踪data_loader
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            y = y.to(self.device)
            y_pred = self.model(x_dict)
            loss = self.criterion(y_pred, y.float())
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                print('epoch:', epoch_i, 'validation: auc:', auc)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  # save best auc model

    def evaluate(self, model, data_loader):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts)

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts
