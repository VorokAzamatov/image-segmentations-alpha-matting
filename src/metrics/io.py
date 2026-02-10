import torch

import os



def save_metrics(metrics_save_dir, metrics_dict):
    filename = 'metrics.pt'

    os.makedirs(metrics_save_dir, exist_ok=True)
    metrics_save_path = os.path.join(metrics_save_dir, filename)
    torch.save(metrics_dict, metrics_save_path)
    if os.path.exists(metrics_save_path):
        print(f"Метрики успешно сохранены в папку {metrics_save_dir} как {filename}")
    else:
        print(f"ОШИБКА: файл {metrics_save_path} не сохранён")