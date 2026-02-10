import torch
import mlflow
import mlflow.pytorch

from torch import nn
from torch import optim

from models.UNet import UNet
from training.loops import run_train
from training.callbacks import EarlyStopping, MLflowLoggerCallback
from metrics.io import save_metrics
from data.loaders import get_loaders
from data.datasets import DUTSdataset, AIM500_dataset
from data.transforms import get_train_transforms, get_val_transforms
from configs.config import *



def main():
    model = UNet(in_ch=IN_CH, num_cl=NUM_CL, base_ch=BASE_CH).to(DEVICE)
    train_transforms = get_train_transforms(image_size=IMAGE_SIZE)
    val_test_transforms = get_val_transforms(image_size=IMAGE_SIZE)


    if RUNNING_TRAIN:
        stage = "pretrain"

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
        mlflow_callback = MLflowLoggerCallback(stage=stage)

        train_loader, val_loader, test_loader = get_loaders(
            dataset_class = DUTSdataset,
            data_path = DUTS_DATA_PATH,
            subset_size = DUTS_SUBSET_SIZE,
            batch_size = BATCH_SIZE,
            train_transforms = train_transforms,
            val_test_transforms = val_test_transforms
        )

        
        with mlflow.start_run(run_name=stage):
            mlflow.log_param("stage", stage)
            mlflow.log_param("dataset", "DUTS")

            mlflow.log_params({
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "base_ch": BASE_CH,
                "img_size": IMAGE_SIZE,
                "optimizer": "Adam",
                "criterion": "BCEWithLogitsLoss",
            })

            model, metrics_dict = run_train(model, optimizer, criterion, EPOCHS, EVERY_N_EP, train_loader, val_loader, lr_scheduler, earlystopping, logger_callback=mlflow_callback, device=DEVICE)
        mlflow.end_run()


        save_metrics(METRICS_SAVE_DIR, metrics_dict)
        mlflow.pytorch.log_model(model, f"{stage}_best_model")

    else:
        model.load_state_dict(torch.load(BEST_MODEL_LOAD_PATH, map_location=DEVICE))
        metrics_dict = torch.load(METRICS_PATH)

        if FINETUNE:
            stage = "finetune"

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
            FT_mlflow_callback = MLflowLoggerCallback(stage=stage)

            FT_train_loader, FT_val_loader, FT_test_loader = get_loaders(
                dataset_class = AIM500_dataset,
                data_path = AIM500_DATA_PATH,
                subset_size = AIM_SUBSET_SIZE,
                batch_size = BATCH_SIZE,
                train_transforms = train_transforms,
                val_test_transforms = val_test_transforms
            )
            
            
            with mlflow.start_run(run_name=stage):
                mlflow.log_param("stage", stage)
                mlflow.log_param("dataset", "AIM-500")

                mlflow.log_params({
                    "epochs": FT_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": FT_LR,
                    "base_ch": BASE_CH,
                    "img_size": IMAGE_SIZE,
                    "optimizer": "Adam",
                    "criterion": "BCEWithLogitsLoss",
                })

                model, metrics_dict = run_train(model, FT_optimizer, FT_criterion, FT_EPOCHS, EVERY_N_EP, FT_train_loader, FT_val_loader, FT_lr_scheduler, FT_earlystopping, logger_callback=FT_mlflow_callback, device=DEVICE)

            save_metrics(FT_METRICS_SAVE_DIR, metrics_dict)
            mlflow.pytorch.log_model(model, f"{stage}_best_model")




if __name__ == "__main__":
    main()