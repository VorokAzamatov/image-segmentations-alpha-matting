import torch
import matplotlib.pyplot as plt

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