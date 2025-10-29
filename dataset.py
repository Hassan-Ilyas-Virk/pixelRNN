# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageCompletionDataset(Dataset):
    def __init__(self, root_dir, image_size=(64, 64)):
        self.root_dir = root_dir
        
        # Corrected paths to match your actual folder structure
        self.occluded_dir = os.path.join(root_dir, 'train', 'occluded_images')
        self.original_dir = os.path.join(root_dir, 'train', 'original_images')
        
        # Find all image files with common extensions
        self.image_files = [f for f in os.listdir(self.occluded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Get the occluded image filename (e.g., "occluded_bed_1.jpg")
        img_name = self.image_files[idx]
        
        # 2. Create the corresponding original image filename (e.g., "bed_1.jpg")
        original_img_name = img_name.replace("occluded_", "")
        
        # 3. Create the full paths for both images
        occluded_path = os.path.join(self.occluded_dir, img_name)
        original_path = os.path.join(self.original_dir, original_img_name) # Use the corrected name
        
        # 4. Open the images
        occluded_img = Image.open(occluded_path).convert('RGB')
        original_img = Image.open(original_path).convert('RGB')
        
        # 5. Apply transformations
        occluded_tensor = self.transform(occluded_img)
        original_tensor = self.transform(original_img)
        
        # 6. Convert the target image to class labels (0-255) for the loss function
        original_labels = (original_tensor * 255).long()
        
        # 7. Return the pair
        return occluded_tensor, original_labels