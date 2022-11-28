from PIL import Image
import uuid
from urllib import request
from urllib.request import urlopen
import os
import encodings
import config as cfg


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