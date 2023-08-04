from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train import train_transform, Dataset


def plot_batch(batch):
    batch_np = batch.numpy()

    for i in range(len(batch_np)):
        # y_channel = batch_np[i, :, :]
        plt.subplot(2, 4, i + 1)  # Change the subplot size according to the batch size (2 rows, 4 columns)
        plt.imshow(batch_np[i])
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)  # Adjust the spacing between subplots
    plt.show()


def main():
    dataset = load_dataset("eugenesiow/Div2k")  # Load the dataset

    train_dataset = Dataset(dataset=dataset['train'], transform=train_transform)

    # PyTorch dataloaders
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=16,
        drop_last=True,
        pin_memory=True,
    )
    for batch in dataloader:
        plot_batch(batch[0])
        break  # Only show the first batch


if __name__ == '__main__':
    main()
