import torch
import matplotlib.pyplot as plt

import os
import random

from torch.utils.data import DataLoader


def visualize_predictions(model, loader, device, epoch=None, n=3):
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
        return tensor * std + mean

    batch = next(iter(loader))
    X = batch['img'][:n].to(device)
    Y = batch['mask'][:n]
    
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(X))

    plt.figure(figsize=(8, 3 * n))
    epoch_str = f"| {epoch} epoch" if epoch is not None else ''
    plt.suptitle(f"UNet: Predictions {epoch_str}", fontsize=16)
    
    for i in range(n):
        prediction = pred[i][0].cpu().numpy()
        mask = Y[i].permute(1, 2, 0)
        img = denormalize(X.cpu())[i].permute(1, 2, 0)
        
        plt.subplot(n, 3, i * 3 + 1)
        plt.title("Image")
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(n, 3, i * 3 + 2)
        plt.title("Ground Truth")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(n, 3, i * 3 + 3)
        plt.title("Prediction")
        plt.imshow(prediction, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_metrics(metrics_save_dir, metrics_dict):
    filename = 'metrics.pt'

    os.makedirs(metrics_save_dir, exist_ok=True)
    metrics_save_path = os.path.join(metrics_save_dir, filename)
    torch.save(metrics_dict, metrics_save_path)
    if os.path.exists(metrics_save_path):
        print(f"Метрики успешно сохранены в папку {metrics_save_dir} как {filename}")
    else:
        print(f"ОШИБКА: файл {metrics_save_path} не сохранён")


def get_loaders(dataset_class, data_path, subset_size, batch_size, train_transforms=None, val_test_transforms=None):
    dataset = dataset_class(data_path, transforms = None)

    pairs = dataset.all_pairs.copy()
    
    if subset_size is not None:
        assert subset_size <= len(pairs)
        pairs = random.sample(pairs, subset_size)

        dataset = dataset_class(data_path, pairs = pairs, transforms = None)

    dataset_len = len(dataset)

    train_len = int(dataset_len * 0.7)
    val_len = int(dataset_len * 0.2)

    train_pairs = pairs[:train_len]
    val_pairs = pairs[train_len:train_len + val_len]
    test_pairs = pairs[train_len + val_len:]

    train_dataset = dataset_class(data_path, transforms=train_transforms, pairs=train_pairs)
    val_dataset = dataset_class(data_path, transforms=val_test_transforms, pairs=val_pairs)
    test_dataset = dataset_class(data_path, transforms=val_test_transforms, pairs=test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader