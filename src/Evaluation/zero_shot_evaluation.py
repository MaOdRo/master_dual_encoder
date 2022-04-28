import tqdm
import config as cfg
import torch
import numpy as np
import clip
import torch.nn.functional as F

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
            


#def eval_clip():
    #print("Scoring clip english accuracy...")
    #accuracy = eval_eng_topk_accuracy(df_wiki_eng_img, eval_data_eng, image_eng_embed)
    #print(f"CLIP accuracy: {round(accuracy * 100, 3)}%")