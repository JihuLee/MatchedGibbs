import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchvision import transforms
from src.models import ResNet18_Encoder

tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_files = sorted(
            [f for f in self.root_dir.glob('*.jpg')], 
            key=lambda x: int(x.stem)
        )
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, str(img_path.stem)

def get_embeddings(data_dir, batch_size=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    model = ResNet18_Encoder(pretrained=True).to(device)
    model.eval()
    
    dataset = CelebADataset(data_dir, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_embeddings = {}
    
    with torch.no_grad():
        for images, img_ids in tqdm(dataloader, desc="Generating embeddings"):
            images = images.to(device)
            embeddings = model(images).cpu().numpy()
            
            for i, img_id in enumerate(img_ids):
                all_embeddings[img_id] = embeddings[i]
    
    return all_embeddings

def save_embeddings(embeddings_dict, output_path):
    
    ids = np.array(list(embeddings_dict.keys()))
    embeddings = np.array([embeddings_dict[img_id] for img_id in ids])
    
    np.savez_compressed(
        output_path,
        ids=ids,
        embeddings=embeddings
    )
    print(f"Saved {len(ids)} embeddings to {output_path}")

def load_embeddings(npz_path):
    
    data = np.load(npz_path)
    ids = data['ids']
    embeddings = data['embeddings']
    return {img_id: emb for img_id, emb in zip(ids, embeddings)}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate ResNet18 embeddings for CelebA dataset')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing CelebA images')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the embeddings (should end with .npz)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for processing')
    args = parser.parse_args()
    
    print(f"Generating embeddings for images in {args.data_dir}")
    embeddings = get_embeddings(args.data_dir, batch_size=args.batch_size)
    save_embeddings(embeddings, args.output_path)
    
    loaded_embeddings = load_embeddings(args.output_path)
    print(f"Successfully generated and saved {len(loaded_embeddings)} embeddings")
    print(f"First image ID: {next(iter(loaded_embeddings))}")
    print(f"Last image ID: {next(reversed(loaded_embeddings))}")

if __name__ == "__main__":
    main()