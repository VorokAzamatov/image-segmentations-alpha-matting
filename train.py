import torch

from torch import nn
from torch import optim

from models.unet import UNet
from training.loops import run_train
from training.callbacks import EarlyStopping
from utils import get_loaders, save_metrics
from datasets.datasets import DUTSdataset, AIM500_dataset, get_train_transforms, get_val_transforms
from configs.config import *



model = UNet(in_ch=IN_CH, num_cl=NUM_CL, base_ch=BASE_CH).to(DEVICE)
train_transforms = get_train_transforms(image_size=512)
val_test_transforms = get_val_transforms(image_size=512)


if RUNNING_TRAIN:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = 'min', 
        factor = FACTOR, 
        patience = LR_SCHEDULER_PATIENCE, 
        threshold = MIN_DELTA, 
        threshold_mode = 'abs'
    )
    earlystopping = EarlyStopping(
        mode = 'min', 
        min_delta = MIN_DELTA, 
        patience = EARLYSTOPPING_PATIENCE,  
         best_model_dir = BEST_MODEL_SAVE_DIR, 
        verbose = True
    )

    train_loader, val_loader, test_loader = get_loaders(
        dataset_class = DUTSdataset,
        data_path = DUTS_DATA_PATH,
        subset_size = DUTS_SUBSET_SIZE,
        batch_size = BATCH_SIZE,
        train_transforms = train_transforms,
        val_test_transforms = val_test_transforms
    )

    model, metrics_dict = run_train(model, optimizer, criterion, EPOCHS, EVERY_N_EP, train_loader, val_loader, lr_scheduler, earlystopping, DEVICE)
    
    save_metrics(METRICS_SAVE_DIR, metrics_dict)
        
else:
    model.load_state_dict(torch.load(BEST_MODEL_LOAD_PATH, map_location=DEVICE))
    metrics_dict = torch.load(METRICS_PATH)

    if FINETUNE:
        FT_optimizer = optim.Adam(model.parameters(), lr=FT_LR)
        FT_criterion = nn.BCEWithLogitsLoss()
        FT_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            FT_optimizer,
            mode = 'min',
            factor = FT_FACTOR,
            patience = FT_LR_SCHEDULER_PATIENCE,
            min_lr = FT_MINLR
        )
        FT_earlystopping = EarlyStopping(
            patience = FT_EARLYSTOPPING_PATIENCE,
            min_delta = FT_MIN_DELTA,
            best_model_dir = FT_BEST_MODEL_SAVE_DIR,
            verbose = True
        )

        FT_train_loader, FT_val_loader, FT_test_loader = get_loaders(
            dataset_class = AIM500_dataset,
            data_path = AIM500_DATA_PATH,
            subset_size = AIM_SUBSET_SIZE,
            batch_size = BATCH_SIZE,
            train_transforms = train_transforms,
            val_test_transforms = val_test_transforms
        )

        model, metrics_dict = run_train(model, FT_optimizer, FT_criterion, FT_EPOCHS, EVERY_N_EP, FT_train_loader, FT_val_loader, FT_lr_scheduler, FT_earlystopping, DEVICE)

        save_metrics(FT_METRICS_SAVE_DIR, metrics_dict)