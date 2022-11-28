import tqdm
from config import config as cfg
import torch
import numpy as np
import clip
import torch.nn.functional as F
import pandas as pd
from helper_functions import load_image, utf8_byte_truncate
from tqdm import tqdm


clip_model, preprocess = clip.load("ViT-B/32", device = cfg.device)

def get_data(name):
    dataset = pd.read_csv('../master_dual_encoder/data/' + name)
    return dataset


#Gesamte image embeddings erstellen
def create_image_embeddings(image_urls, model):
    num_batches = int(np.ceil(len(image_urls) / cfg.batch_size))
    img_embed = []
    
    for idx in tqdm(range(num_batches)):
        start_idx = idx * cfg.batch_size
        end_idx = start_idx + cfg.batch_size
        current_image_urls = image_urls[start_idx:end_idx]
        
        img_input = torch.stack([preprocess(load_image(img)).to(cfg.device) for img in current_image_urls])
        
        with torch.no_grad():
            current_image_embed = model.encode_image(img_input.to(cfg.device)).float()
        
        img_embed.append(current_image_embed)
        del img_input
        del current_image_embed
        torch.cuda.empty_cache()
    img_embeds = torch.cat(img_embed)
    return img_embeds


def find_query_match(image_urls, img_emb, queries, model, n = 5):
    query_tok = clip.tokenize(queries).to(cfg.device)
    
    with torch.no_grad():
            cap_emb = model.encode_text(query_tok).float().to(cfg.device)
            
    image_embeddings_n = F.normalize(img_emb, p=2, dim=-1)
    text_embeddings_n = F.normalize(cap_emb, p=2, dim=-1)
    
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    
    matches = [[image_urls[idx] for idx in results] for results in indices]

    return matches



def eval_eng_topk_accuracy(image_urls, eval_data, img_embed, n = cfg.n_faktor):
    hits = 0
    num_batches = int(np.ceil(len(image_urls) / cfg.batch_size))
    
    for idx in tqdm(range(num_batches)):
        start_idx = idx * cfg.batch_size
        end_idx = start_idx + cfg.batch_size
        current_image_urls = image_urls[start_idx:end_idx]
        
        queries = [
            eval_data[url] for url in current_image_urls 
        ]
        
        result = find_query_match(image_urls, img_embed, queries, n)

        hits += sum(
            [
                url in matches
                for (url, matches) in list(zip(current_image_urls, result))
            ]
        )
        
    return hits / len(image_urls)
            


def eval_clip():
    df = get_data('wikidata.csv')
    df_wiki_img = df['image_url'].tolist()
    df_wiki_cap = df['caption_title_and_reference_description'].tolist()

    #Truncation auf CLIP content length
    for i, caption in enumerate(df_wiki_cap):
        if len(caption) > cfg.context_length:
            df_wiki_cap[i] = caption[:cfg.context_length]

    #truncate alle utf-8 strings
    for i, caption in enumerate(df_wiki_cap):
        if len(caption.encode('utf-8')) >= cfg.content_length_unicode:
            df_wiki_cap[i] = utf8_byte_truncate(caption, cfg.content_length_unicode)


    eval_data = dict(zip(df_wiki_img, df_wiki_cap))
    image_embed = create_image_embeddings(df_wiki_img, clip_model)

    print("Scoring clip english accuracy...")
    accuracy = eval_eng_topk_accuracy(df_wiki_img, eval_data, image_embed)
    print(f"CLIP accuracy: {round(accuracy * 100, 3)}%")

eval_clip()