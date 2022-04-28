import torch
from src.helper_functions import load_image
import random


def make_english_data():
    pass


class WikiClipDataset(torch.utils.data.Dataset):
    """Klasse zur erstellung des Datensatzes. Erstellt Listen aus den
        Bildern und Captions, transformiert diese und gibt sie als Tensoren wieder aus

    Args:
        torch (utils.data.Dataset): Datensatz Klasse
    """
    def __init__(self, df, transform_img = None, transform_text = None):

        self.images = df["image_url"].tolist()
        self.captions = df['caption_title_and_reference_description'].tolist()
        self.transform_img = transform_img
        self.transform_text = transform_text
        

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #Das Bild für CLIP 
        image_raw = load_image(self.images[idx])
        #falls image immernoch 404 neue Indexnummer nehmen
        while image_raw == None:
            idx = random.randint(0, len(self.images)-1)
            image_raw = load_image(self.images[idx])
        image = self.transform_img(image_raw)
        #Caption für CLIP 
        caption_raw = self.captions[idx]
        caption = self.transform_text(caption_raw)
        
        return image, caption
