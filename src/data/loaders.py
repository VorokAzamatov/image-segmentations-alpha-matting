import random

from torch.utils.data import DataLoader

import random

from torch.utils.data import DataLoader

def get_loaders(dataset_class, data_path, batch_size, subset_size=None, train_transforms=None, val_test_transforms=None):
    dataset = dataset_class(data_path, transforms = None)

    pairs = dataset.all_pairs.copy()

    random.shuffle(pairs)
    
    if subset_size is not None:
        assert subset_size <= len(pairs)
        pairs = pairs[:subset_size]
    
    dataset_len = len(pairs)

    print(dataset_len)

    train_len = int(dataset_len * 0.8)
    val_len = int(dataset_len * 0.1)

    train_pairs = pairs[:train_len]
    val_pairs = pairs[train_len:train_len + val_len]
    test_pairs = pairs[train_len + val_len:]
    
    train_dataset = dataset_class(data_path, transforms=train_transforms, pairs=train_pairs)
    val_dataset = dataset_class(data_path, transforms=val_test_transforms, pairs=val_pairs)
    test_dataset = dataset_class(data_path, transforms=val_test_transforms, pairs=test_pairs)

    print(f"train_dataset len: {len(train_dataset)}")
    print(f"test_dataset len: {len(test_dataset)}")
    print(f"val_dataset len: {len(val_dataset)}")
    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader