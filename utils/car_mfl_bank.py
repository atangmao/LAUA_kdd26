# utils/car_mfl_bank.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def build_public_bank(public_loader, device, normalize=True):
    """
    从 public_loader 构建特征库（cached CLIP features 场景下非常快）
    public batch: {'image':[B,512], 'text':[B,512], 'sample_idx':[B]}
    Returns:
      bank = {
        'img': [N,D] float32 on device
        'txt': [N,D] float32 on device
        'sample_idx': [N] long on cpu
      }
    """
    imgs, txts, idxs = [], [], []
    for batch in public_loader:
        imgs.append(batch['image'])
        txts.append(batch['text'])
        idxs.append(batch['sample_idx'])
    img = torch.cat(imgs, dim=0).float()
    txt = torch.cat(txts, dim=0).float()
    sample_idx = torch.cat(idxs, dim=0).long()

    if normalize:
        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)

    return {
        'img': img.to(device, non_blocking=True),
        'txt': txt.to(device, non_blocking=True),
        'sample_idx': sample_idx,  # keep cpu
    }

@torch.no_grad()
def retrieve_top1(query, bank, modality='img'):
    """
    query: [B,D] on device (already normalized if bank normalized)
    modality='img' => search in bank['img']
    modality='txt' => search in bank['txt']
    return: indices [B] (row index in bank)
    """
    key = 'img' if modality == 'img' else 'txt'
    sim = query @ bank[key].T  # [B,N]
    idx = torch.argmax(sim, dim=1)
    return idx