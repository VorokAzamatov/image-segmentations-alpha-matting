
import torch

from tqdm import tqdm

from utils.vizualization import visualize_predictions 


def train_epoch(model, train_loop, optimizer, criterion, epochs, epoch, device, metric_funcs: list):
    running_loss = []
    running_metrics = [ [] for i in range(len(metric_funcs))]


    model.train()
    for batch in train_loop:
        x = batch['img'].to(device)
        y = batch['mask'].to(device)
        
        pred = model(x)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        running_loss.append(loss.item())
        for i, metric_func in enumerate(metric_funcs):
            metric_value = metric_func(pred, y)
            running_metrics[i].append(metric_value.item())


        
        mean_loss = sum(running_loss) / len(running_loss)
        mean_metrics = [sum(m)/len(m) for m in running_metrics]
        
        metrics_str  =  " | ".join([ f"{metric_func.__name__}: {mean_metrics[i]:.4f}"   for i, metric_func in enumerate(metric_funcs) ])
        train_loop.set_description(f"[{epoch}/{epochs}] train_loss: {mean_loss:.4f} | {metrics_str}")


    return mean_loss, { f"train_{metric_func.__name__.replace('_metric', '')}": mean_metrics[i] for i, metric_func in enumerate(metric_funcs) }


def eval_epoch(model, loader, criterion, device, metric_funcs: list):
    running_loss = []
    running_metrics = [ [] for i in range(len(metric_funcs))]

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch['img'].to(device)
            y = batch['mask'].to(device)

            pred = model(x)
            
            loss = criterion(pred, y)
            
            running_loss.append(loss.item())
            for i, metric_func in enumerate(metric_funcs):
                metric_value = metric_func(pred, y)
                running_metrics[i].append(metric_value.item())
        
        mean_loss = sum(running_loss) / len(running_loss)
        mean_metrics = [sum(m)/len(m) for m in running_metrics]

    return mean_loss, { f"val_{metric_func.__name__.replace('_metric', '')}": mean_metrics[i]   for i, metric_func in enumerate(metric_funcs) }


def run_train(model, optimizer, criterion, epochs, every_n_ep, train_loader, val_loader, lr_scheduler, earlystopping, device, metric_funcs: list, logger_callback=None):
    mean_train_loss_list = []
    mean_val_loss_list = []
    metrics_history = None
    lr_list = []
    
    for epoch in range(1, epochs+1):
        train_loop = tqdm(train_loader, leave = False)
        
        mean_train_loss, train_metrics_dict = train_epoch(model, train_loop, optimizer, criterion, epochs, epoch, metric_funcs= metric_funcs, device=device)
        
        mean_val_loss, val_metrics_dict = eval_epoch(model, val_loader, criterion, metric_funcs= metric_funcs, device=device)

        metrics_dict = train_metrics_dict | val_metrics_dict 
        

        earlystopping.step(mean_val_loss, model, epoch)
        lr_scheduler.step(mean_val_loss)
        lr = optimizer.param_groups[0]['lr']

        if logger_callback is not None:
            logger_callback.log_epoch(
                metrics_dict = metrics_dict,
                mean_train_loss = mean_train_loss, 
                mean_val_loss = mean_val_loss, 
                epoch = epoch,
                lr = lr
            )
            

        mean_train_loss_list.append(mean_train_loss)
        mean_val_loss_list.append(mean_val_loss)
        lr_list.append(lr)


        if metrics_history is None:
            metrics_history = { k: [] for k in metrics_dict.keys() }

        for k, v in metrics_dict.items():
            metrics_history[k].append(v)


        metrics_str = " | ".join([ f"{metric_name}: {metric_value:.4f}" for metric_name, metric_value in metrics_dict.items() ])
        print(f"[{epoch}/{epochs}] train_loss: {mean_train_loss:.4f} | {metrics_str} | lr: {lr} ||| best_val_loss: {earlystopping.best_value:.6f}")


        if epoch == 1 or epoch % every_n_ep == 0:
            visualize_predictions(model, val_loader, device, epoch, n=2)
        
        if earlystopping.should_stop:
            print(f"Stopped on epoch: {epoch}")
            break
        

    model = earlystopping.get_best_model(model)
    metrics_history_dict = {
        'train_loss_list': mean_train_loss_list,
        'val_loss_list': mean_val_loss_list,
        'lr_list': lr_list,
        **metrics_history
    }

    return model, metrics_history_dict