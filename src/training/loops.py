
import torch

from tqdm import tqdm

from metrics import mse_metric
from utils import visualize_predictions


def train_epoch(model, train_loop, optimizer, criterion, epochs, epoch, device):
    running_train_loss = []
    running_train_mse = []

    model.train()
    for batch in train_loop:
        x_train = batch['img'].to(device)
        y_train = batch['mask'].to(device)
        
        pred = model(x_train)

        loss = criterion(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_mse = mse_metric(pred, y_train)

        running_train_loss.append(loss.item())
        running_train_mse.append(train_mse.item())

        mean_train_loss = sum(running_train_loss) / len(running_train_loss)
        mean_train_mse = sum(running_train_mse) / len(running_train_mse)

        train_loop.set_description(f"[{epoch}/{epochs}] train_loss: {mean_train_loss:.4f} | train_mse: {mean_train_mse:.4f}")

    return mean_train_loss, mean_train_mse


def eval_epoch(model, loader, criterion, device):
    running_loss = []
    running_mse = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch['img'].to(device)
            y = batch['mask'].to(device)

            pred = model(x)
            
            loss = criterion(pred, y)
            
            mse = mse_metric(pred, y)

            running_loss.append(loss.item())
            running_mse.append(mse.item())
        
        mean_loss = sum(running_loss) / len(running_loss)
        mean_mse = sum(running_mse) / len(running_mse)

    return mean_loss, mean_mse


def run_train(model, optimizer, criterion, epochs, every_n_ep, train_loader, val_loader, lr_scheduler, earlystopping, device, logger_callback=None):
    mean_train_loss_list = []
    mean_train_mse_list = []
    mean_val_loss_list = []
    mean_val_mse_list = []
    lr_list = []
    
    for epoch in range(1, epochs+1):
        train_loop = tqdm(train_loader, leave = False)
        
        mean_train_loss, mean_train_mse = train_epoch(model, train_loop, optimizer, criterion, epochs, epoch, device=device)
        
        mean_val_loss, mean_val_mse = eval_epoch(model, val_loader, criterion, device=device)

        
        earlystopping.step(mean_val_loss, model, epoch)
        lr_scheduler.step(mean_val_loss)
        lr = lr_scheduler._last_lr[0]

        if logger_callback is not None:
            logger_callback.log_epoch(epoch, mean_train_mse, mean_train_loss, mean_val_mse, mean_val_loss, lr)
            

        mean_train_loss_list.append(mean_train_loss)
        mean_train_mse_list.append(mean_train_mse)
        mean_val_loss_list.append(mean_val_loss)
        mean_val_mse_list.append(mean_val_mse)
        lr_list.append(lr)

        print(f"[{epoch}/{epochs}] train_loss: {mean_train_loss:.4f} | train_mse: {mean_train_mse:.4f} | val_loss: {mean_val_loss:.4f} | val_mse: {mean_val_mse:.4f} | lr: {lr} ||| best_val_loss: {earlystopping.best_value:.6f}")

        if epoch == 1 or epoch % every_n_ep == 0:
            visualize_predictions(model, val_loader, device, epoch, n=2)
        
        if earlystopping.should_stop == True:
            print(f"Stopped on epoch: {epoch}")
            break
        
    model = earlystopping.get_best_model(model)
    metrics_dict = {
        'train_loss_list': mean_train_loss_list,
        'train_mse_list': mean_train_mse_list,
        'val_loss_list': mean_val_loss_list,
        'val_mse_list': mean_val_mse_list,
        'lr_list': lr_list
    }

    return model, metrics_dict