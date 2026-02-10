import random

from torch.utils.data import DataLoader

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