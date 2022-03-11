from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import DnCNN
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def average(lst):
    return sum(lst) / len(lst)


class MRIDataset(Dataset):
    def __init__(self):
        self.xHat = torch.from_numpy(np.load('xHat.npy'))
        self.grndTrth = torch.from_numpy(np.load('grndTrth.npy'))
        self.n_samples = self.xHat.shape[0]

    def __getitem__(self, index):
        return self.xHat[index].float(), self.grndTrth[index].float()

    def __len__(self):
        return self.n_samples


device = torch.device('cpu')
model = DnCNN()
model.load_state_dict(torch.load("trained_model", map_location=device))
model.eval()

batch_size = 4
dataset = MRIDataset()
testing_data = torch.utils.data.Subset(dataset, range(261, 360))
testing_loader = DataLoader(dataset=testing_data, batch_size=batch_size)

num_images = dataset.n_samples

xs = np.empty(shape=(num_images, 256, 232))
outputs = np.empty(shape=(num_images, 256, 232))
grndTrths = np.empty(shape=(num_images, 256, 232))
psnr_before = []
psnr_after = []
ssim_before = []
ssim_after = []

for i, (x, grndTrth) in enumerate(testing_loader):
    output = model(x)

    grndTrths[i] = grndTrth[0][0].detach().numpy()
    outputs[i] = output[0][0].detach().numpy()
    xs[i] = x[0][0].detach().numpy()

    psnr_before.append(psnr(grndTrths[i], xs[i], data_range=grndTrths[i].max() - grndTrths[i].min()))
    psnr_after.append(psnr(grndTrths[i], outputs[i], data_range=grndTrths[i].max() - grndTrths[i].min()))

    ssim_before.append(ssim(grndTrths[i], xs[i], data_range=grndTrths[i].max() - grndTrths[i].min()))
    ssim_after.append(ssim(grndTrths[i], outputs[i], data_range=grndTrths[i].max() - grndTrths[i].min()))

print(f'PSNR Before Average: {average(psnr_before)}')
print(f'PSNR After Average: {average(psnr_after)}')
print(f'SSIM Before Average: {average(ssim_before)}')
print(f'SSIM After Average: {average(ssim_after)}')

for i in range(num_images):
    fig = plt.figure(figsize=(6, 8))
    fig.canvas.manager.set_window_title(f'Test {i+1}')

    grndTrth = grndTrths[i]
    output = outputs[i]
    x = xs[i]

    fig.add_subplot(3, 3, 1)
    plt.imshow(x, cmap="gray")
    plt.axis('off')
    plt.title('Input (xHat):')

    fig.add_subplot(3, 3, 2)
    plt.imshow(output, cmap="gray")
    plt.axis('off')
    plt.title('Prediction:')

    fig.add_subplot(3, 3, 3)
    plt.imshow(grndTrth, cmap="gray")
    plt.axis('off')
    plt.title('Ground Truth (x):')

    fig.add_subplot(3, 3, 7)
    plt.title(f'PSNR Before: {psnr_before[i]:.4f} \n PSNR After: {psnr_after[i]:.4f}')
    plt.axis('off')

    fig.add_subplot(3, 3, 9)
    plt.title(f'SSIM Before: {ssim_before[i]:.4f} \n SSIM After: {ssim_after[i]:.4f}')
    plt.axis('off')

    plt.show()
    plt.close(fig)
