import multiprocessing
import torch

class config:
    #allgemein
    seed = 42
    batch_size = 64
    epochs = 5
    num_workers = multiprocessing.cpu_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #learning rates
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    head_lr = 1e-3
    #optimizer = optim.Adam(clip_model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 
    #Parameter aus dem Paper, lr für das finetunen ein bisschen geringer gesetzt

    #CLIP
    clip_visual_model = 'RN50x4'
    clip_embed_dim = 640
    embedding_dim = 512

    #TextEncoder
    text_encoder_model = "xlm-roberta-base"
    text_embedding = 768
    context_length = 77
    content_length_unicode = 60


    #Projection Head
    num_layer = 3
    dropout = 0.5
    projection_dim = 256

    n_faktor = 20