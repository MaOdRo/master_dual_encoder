from PIL import Image
import uuid
from urllib import request
from urllib.request import urlopen
import os
import encodings
import config as cfg
from config import config as cfg
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from typing import Dict, Tuple, List
import random

def load_image(link):
    """Laden des Bildes aus dem übergebenen Link und Ausgabe als PIL Image

    Args:
        link (string): die wiki-url des Bildes

    Returns:
        PIL Image: Das geladene Bild
    """    
    try:
        URL = link#.rsplit(';', 1)[1]
        filename = str(uuid.uuid4())
        path = f'./data/images/{filename}'
        req = request.Request(URL)
        req.add_header('User-Agent', 'User-bot-abc')#,'Connection', 'close')
        response = request.urlopen(req)
        
        with open(path, 'wb') as f:
            f.write(response.read())
        
        image = Image.open(path).convert("RGB")
        
        os.remove(path)
        
        return image
    
    except Exception as e:
        print(e)
        return None



incr_encoder = encodings.search_function('utf8').incrementalencoder()

def utf8_byte_truncate(text, max_bytes):
    """Abschneiden eines texts auf die bestimmte Länge

    Args:
        text (string): übergebenes caption
        max_bytes (integer): abschneiden auf diese länge

    Returns:
        string: neuer kürzerer text
        Quelle:
        https://stackoverflow.com/questions/13665001/python-truncating-international-string?lq=1
    """    
    byte_len = 0
    incr_encoder.reset()
    for index,ch in enumerate(text):
        byte_len += len(incr_encoder.encode(ch))
        if byte_len > max_bytes:
            break
    else:
        return text
    return text[:index]



def truncate_nrm(cap_ref):
    """Abschneiden aller captions auf die bestimmte Länge

    Args:
        cap_ref (list): liste aller captions
    """    
    for i, caption in enumerate(cap_ref):
        if len(caption) > cfg.context_length:
            cap_ref[i] = caption[:cfg.context_length]


def truncate_utf8(cap_ref):
    """Abschneiden aller utf-8 captions auf die bestimmte Länge

    Args:
        cap_ref (list): liste aller captions
    """
    for i, caption in enumerate(cap_ref):
        if len(caption.encode('utf-8')) >= cfg.content_length_unicode:
            cap_ref[i] = utf8_byte_truncate(caption, cfg.content_length_unicode)


"""
Funktionen für die Berechnung der Verlustfunktion
"""
def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0

def metrics(similarity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    cap_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, cap_acc


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



class Tokenizer:
    """Autotokenizer aus der Huggingface Bibliothek
        Start- und Endtokens werden hinzugefügt und max_length auf 77 gesetzt
    Returns:
        input_ids und entsprechende attention_mask
    """
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, caption: str) -> AutoTokenizer:
        return self.tokenizer(
            caption,
            add_special_tokens = True,#<s> und </s> hinzufügen
            max_length = cfg.context_length,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'pt',
        )

    def decode(self, x: Dict[str, torch.LongTensor]):
        return [self.tokenizer.decode(sentence[:sentence_len]) for sentence, sentence_len in
                zip(x['input_ids'], x['attention_mask'].sum(axis=-1))]



class ProjectionHead(nn.Module):
    """Die übergebenen Vektoren werden auf die Gleiche Dimension gebracht
    Args:
        nn (neural network Module): Standart Module
    """    
    def __init__(self, dim_in, dim_out, p = cfg.dropout):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out, bias=False)
        self.linear2 = nn.Linear(dim_out, dim_out, bias=False)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds



class ImageEncoder(nn.Module):
    """Die Basis bleibt der Image Encoder von CLIP

    Args:
        nn (neural network Module): Standart Module
    """
    def __init__(self, clip_vision_model, dim_in, dim_out):
        super().__init__()
        model = clip_vision_model
        self.model = model
        self.projection = ProjectionHead(dim_in, dim_out)
        for p in self.model.parameters():#backbone wird gefreezed
            p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.model(x))
        projection_len = torch.norm(projected_vec, dim = -1, keepdim = True)
        return projected_vec / projection_len



class TextEncoder(nn.Module):
    """Als Text Encoder wird xlm-roberta-base benutzt

    Args:
        nn (neural network Module): Standart Module
    """
    def __init__(self, text_encoder_model, dim_out):
        super().__init__()
        self.model = AutoModel.from_pretrained(text_encoder_model)
        self.projection = ProjectionHead(cfg.text_embedding, dim_out)
        for p in self.model.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        out = self.model(**x)[0]
        out = out[:, 0, :] #</s> token
        projected_vec = self.projection(out)

        projection_len = torch.norm(projected_vec, dim = -1, keepdim = True)
        return projected_vec / projection_len



class WikiCLIPMultilingual(pl.LightningModule):
    """Das Modell WikiCLIPMultilingual wird hier erstellt

    Args:
        pl (pl network Module): Pytorch Lighting Standart Module
    """
    def __init__(self, 
                 clip_model,
                 transform_img,
                 tokenizer,
                 text_model_name,
                 clip_embed_dim = cfg.clip_embed_dim,
                 embed_dim = cfg.embedding_dim,
                ):
        super().__init__()
        self.clip_model = clip_model
        self.transform_img = transform_img
        self.tokenizer = tokenizer
        self.image_encoder = ImageEncoder(
            clip_model.visual,
            clip_embed_dim,
            embed_dim,
        )
        self.text_encoder = TextEncoder(text_model_name, embed_dim)
        self.save_hyperparameters()
        #self.transformer = transformers.AutoModel.from_pretrained(model_name)
    
    def common_step(self, batch: Tuple[torch.Tensor, List[str]]) -> torch.Tensor:
        images, text = batch
        text = {k: torch.squeeze(v, 1).to(cfg.device) for k, v in text.items()}

        image_embed = self.image_encoder(images)
        caption_embed = self.text_encoder(text)
        similarity = caption_embed @ image_embed.T

        loss = clip_loss(similarity)
        img_acc, text_acc = metrics(similarity)
        return loss, img_acc, text_acc

    def training_step(self, batch: Tuple[torch.Tensor, List[str]], *args: list
                     ) -> torch.Tensor:
        loss, img_acc, text_acc = self.common_step(batch)
        self.log('training_loss', loss, on_step=True)
        self.log('training_img_acc', img_acc, on_step=True, prog_bar=True)
        self.log('training_text_acc', text_acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[str]], *args: list
                       ) -> torch.Tensor:
        loss, img_acc, text_acc = self.common_step(batch)
        self.log('validation_loss', loss, on_step=True)
        self.log('validation_img_acc', img_acc, on_step=True, prog_bar=True)
        self.log('validation_text_acc', text_acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        vision_params = {'params': self.image_encoder.projection.parameters(), 'lr': cfg.image_encoder_lr}
        text_params = {'params': self.text_encoder.projection.parameters() , 'lr': cfg.text_encoder_lr}
        optimizer = torch.optim.Adam([vision_params, text_params])
        return optimizer