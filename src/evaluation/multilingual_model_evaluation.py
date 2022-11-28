import tqdm
from config import config as cfg
import torch
import numpy as np
import clip
import torch.nn.functional as F
from helper_functions import *
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm


MODEL_PATH = "../master_dual_encoder/src/model/step15clip.ckpt"

clip_model_multi, compose = clip.load(cfg.clip_visual_model, device = cfg.device)
tokenizer = Tokenizer(AutoTokenizer.from_pretrained(cfg.text_encoder_model))
multi_clip_45_epoch_model = WikiCLIPMultilingual.load_from_checkpoint(MODEL_PATH, clip_model = clip_model_multi, transform_img = compose, tokenizer = tokenizer, text_model_name = cfg.text_encoder_model)
text_model_45 = multi_clip_45_epoch_model.text_encoder.to(cfg.device)
image_model_45 = multi_clip_45_epoch_model.image_encoder.to(cfg.device)



def get_data(name):
    dataset = pd.read_csv('../master_dual_encoder/data/' + name)
    return dataset


def create_own_imgage_embeddings(img_list, image_model):
  num_batches = int(np.ceil(len(img_list) / cfg.batch_size))
  image_embeds = []

  for idx in tqdm(range(num_batches)):
    start_idx = idx * cfg.batch_size
    end_idx = start_idx + cfg.batch_size
    current_image_urls = img_list[start_idx:end_idx]

    #Bilder für CLIP transformieren
    img_input = [compose(load_image(img)).unsqueeze(0).cuda() for img in current_image_urls]

    with torch.cuda.amp.autocast():#Automatic Mixed Precision package
          with torch.no_grad():
              for img in img_input:
                image_embeds.append(image_model(img.to(cfg.device)))
    del img_input
    torch.cuda.empty_cache()
    image_embed = torch.cat(image_embeds)
  return image_embed


def find_own_query_match(text_model, image_url, img_emb, queries, n = 5):
    query_tok = tokenizer(queries)
    tokens = {k: v.to(cfg.device) for k, v in query_tok.items()}
    
    with torch.no_grad():
            cap_emb = text_model(tokens)
            
    image_embeddings_n = F.normalize(img_emb, p=2, dim=-1)
    text_embeddings_n = F.normalize(cap_emb, p=2, dim=-1)
    
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n)
    matches = [[image_url[idx] for idx in results] for results in indices]

    return matches



def eval_45_topk_accuracy(image_urls, eval_data, img_embed, n = cfg.n_faktor):
    hits = 0
    num_batches = int(np.ceil(len(image_urls) / cfg.batch_size))
    
    for idx in tqdm(range(num_batches)):
        start_idx = idx * cfg.batch_size
        end_idx = start_idx + cfg.batch_size
        current_image_urls = image_urls[start_idx:end_idx]
        
        queries = [
            eval_data[url] for url in current_image_urls 
        ]
        result = find_own_query_match(text_model_45, image_urls, img_embed, queries, n)
        
        hits += sum(
            [
                url in matches
                for (url, matches) in list(zip(current_image_urls, result))
            ]
        )
        
    return hits / len(image_urls)


def eval_multi_clip():
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
    image_embed = create_own_imgage_embeddings(df_wiki_img, image_model_45)

    print("Scoring 25-Epoch model multilingual accuracy...")
    accuracy = eval_45_topk_accuracy(df_wiki_img, eval_data, image_embed)
    print(f"CLIP accuracy: {round(accuracy * 100, 3)}%")

eval_multi_clip()