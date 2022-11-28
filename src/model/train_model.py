from config import config as cfg
from pathlib import Path
import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import random_split, DataLoader
from multilingual_clip_wiki import *
import clip
import pandas as pd


def get_data(name):
    dataset = pd.read_csv('../master_dual_encoder/data/' + name)
    return dataset


def create_dl(df, image_transform, tokenizer):
    """Funktion für die Erstellung der Trainings- und Evaluationsdataloadern

    Args:
        df (dataset): übergebener Datensatz
        image_transform (preprocess funktion): preprocessfunktion für die Umwandlung der Bilder
        tokenizer (Klasse Tokenizer): Tokenizer für die Umwandlung der Captions

    Returns:
        train_dl, valid_dl: Gibt die erstellten Dataloader zurück
    """
    
    wikiData = WikiClipDataset(df, image_transform, tokenizer)
    train_len = int(0.7*len(wikiData))
    train_data, valid_data = random_split(wikiData, [train_len, len(wikiData) - train_len], generator = torch.Generator().manual_seed(cfg.seed))

    train_dl = DataLoader(
        train_data,
        cfg.batch_size,
        pin_memory = True,
        shuffle = True,
        num_workers = cfg.num_workers,
        drop_last = True
    )

    valid_dl = DataLoader(
        valid_data,
        cfg.batch_size,
        pin_memory = True,
        shuffle = False,
        num_workers = cfg.num_workers,
        drop_last = False
    )
    
    return train_dl, valid_dl



def train(max_epochs):
    """Trainingsfunktion

    Args:
        max_epochs (Integer): Anzahl der zu trainierenden Epochen
    """
    
    #CLIP-Modell und den Tokenizer instantiieren
    clip_model, compose = clip.load(cfg.clip_visual_model, device = cfg.device, jit = False)
    tokenizer = Tokenizer(AutoTokenizer.from_pretrained(cfg.text_encoder_model))

    #Das eigene Modell erstellen
    model = WikiCLIPMultilingual(
        clip_model = clip_model,
        transform_img = compose,
        tokenizer = tokenizer,
        text_model_name = cfg.text_encoder_model,
        clip_embed_dim = cfg.clip_embed_dim,
        embed_dim = cfg.embedding_dim,
    )

    #Trainer erstellen
    trainer = pl.Trainer(
        max_epochs = max_epochs,
        deterministic = True, #wegen seed
        gpus = torch.cuda.device_count(),
        gradient_clip_val = 1.0,
        accelerator = "auto",
        progress_bar_refresh_rate = 20,
        precision = 16,
    )

    #Data Loader erstellen
    train_dl, valid_dl = create_dl(
        get_data('wikidata.csv'),
        image_transform = compose,
        tokenizer = tokenizer
    )

    #Fit
    trainer.fit(
        model,
        train_dl,
        valid_dl
    )

    trainer.save_checkpoint("clip_multi_model_step.ckpt")


train(cfg.epochs)