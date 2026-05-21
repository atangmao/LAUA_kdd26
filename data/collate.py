import torch
import clip

_clip_model = None
_clip_device = None

def get_clip_model(device='cuda'):
    global _clip_model, _clip_device
    if _clip_model is None:
        _clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        _clip_model.eval()
        _clip_device = device
    return _clip_model

def flickr_collate_fn(batch):
    images = []
    texts = []
    ids = []

    for item in batch:
        images.append(item['image'])
        texts.append(item['text'])
        ids.append(item['id'])

    images = torch.stack(images, dim=0)  # 仍然是 CPU tensor

    clip_model = get_clip_model('cuda')
    with torch.no_grad():
        #
        images = images.to(_clip_device)
        image_features = clip_model.encode_image(images).float()   # GPU tensor

    #
    try:
        text_tokens = clip.tokenize(texts, truncate=True)
    except Exception:
        texts = [t if t else "" for t in texts]
        text_tokens = clip.tokenize(texts, truncate=True)

    with torch.no_grad():
        text_tokens = text_tokens.to(_clip_device)
        text_features = clip_model.encode_text(text_tokens).float()  # GPU tensor

    ids = torch.tensor(ids, dtype=torch.long)

    sample_idx = torch.tensor([item['sample_idx'] for item in batch], dtype=torch.long)
    return {'image': image_features, 'text': text_features, 'id': ids, 'sample_idx': sample_idx}

def flickr_collate_cached_fn(batch):
    images = torch.stack([x['image'] for x in batch], dim=0)
    texts  = torch.stack([x['text'] for x in batch], dim=0)
    ids    = torch.tensor([x['id'] for x in batch], dtype=torch.long)

    sample_idx = torch.tensor([x['sample_idx'] for x in batch], dtype=torch.long)
    return {'image': images, 'text': texts, 'id': ids, 'sample_idx': sample_idx}