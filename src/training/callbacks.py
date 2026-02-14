import torch
import mlflow

import os

from copy import deepcopy


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0.001, patience=10, best_model_dir=None, verbose=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.best_model_dir = best_model_dir

        self.best_value = float('inf')
        self.best_model = None

        self.counter = 0
        self.should_stop = False

    def step(self, metric_value, model, epoch):
        improved = metric_value < self.best_value - self.min_delta if self.mode == 'min' else metric_value > self.best_value + self.min_delta
        
        if improved:
            self.counter = 0
            self.best_value = metric_value

            if model is not None and self.best_model_dir is not None:
                self.best_model = deepcopy(model.state_dict())
                os.makedirs(self.best_model_dir, exist_ok=True)

                epoch_str = f"_{epoch}epoch" if epoch else ''
                best_model_name = f"best_model{epoch_str}.pt"
                save_path = os.path.join(self.best_model_dir, best_model_name)
                
                torch.save(self.best_model, save_path)

                if self.verbose:
                    if os.path.exists(save_path):
                        print(f"Лучшая модель успешно сохранена в папку {self.best_model_dir} как {best_model_name}")
                    else:
                        print(f"ОШИБКА: файл {save_path} не сохранён")
        else:
            self.counter += 1

            if self.counter == self.patience:
                self.should_stop = True

    def get_best_model(self, model):
        if self.best_model is not None:
            model.load_state_dict(self.best_model)
        return model
    

class MLflowLoggerCallback(object):
    def __init__(self, stage):
        self.stage = stage

    def log_epoch(self, **kwargs):
        mlflow.log_metric(f"{self.stage}_train_loss", kwargs['mean_train_loss'], step=kwargs['epoch'])
        mlflow.log_metric(f"{self.stage}_val_loss", kwargs['mean_val_loss'], step=kwargs['epoch'])
        mlflow.log_metric(f"{self.stage}_lr", kwargs['lr'], step=kwargs['epoch'])

        for metric_name, metric_values in kwargs['metrics_dict'].items():
            mlflow.log_metric(f"{self.stage}_{metric_name}", metric_values, step=kwargs['epoch'])