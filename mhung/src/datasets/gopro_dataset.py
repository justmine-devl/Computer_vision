import pandas as pd
import cv2

class GoProDataset:
    def __init__(self, csv_file, split='test'):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        blur_path = row['blur_path']
        sharp_path = row['sharp_path']
        image_id = row['image_id']
        
        blur_img = cv2.imread(blur_path)
        sharp_img = cv2.imread(sharp_path)
        
        if blur_img is None:
            raise ValueError(f"Could not read {blur_path}")
        if sharp_img is None:
            raise ValueError(f"Could not read {sharp_path}")
            
        return {'image_id': image_id, 'blur': blur_img, 'sharp': sharp_img}
