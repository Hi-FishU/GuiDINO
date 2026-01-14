from model.wrapper import MedTokenSegLightningModule
from model.swinunet import SwinUnet
from data.segmentation import (discover_drive_samples, discover_kvasir_samples)

def main():
    # Example usage of the imported classes and functions
    model = SwinUnet(img_size=784, patch_size=4, in_chans=3, num_classes=1)
    criterion = torch.nn.BCEWithLogitsLoss()
    lightning_module = MedTokenSegLightningModule(model=model, criterion=criterion)

    drive_samples = discover_drive_samples("/path/to/drive", split="training", use_manual=True)
    kvasir_samples = discover_kvasir_samples("/path/to/kvasir")

    print(f"Discovered {len(drive_samples)} DRIVE samples.")
    print(f"Discovered {len(kvasir_samples)} Kvasir samples.")