import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

from data.flickr_dataset import Flickr30KDataset  #


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = "./data"  #
    cached_dir = "/home/userdata/data/flicker30k_clip/"
    os.makedirs(cached_dir, exist_ok=True)

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    #
    ds = Flickr30KDataset(root_dir=data_root, split="all", transform=preprocess, cached_feature_dir=None)

    #
    text_features = torch.empty((len(ds), 512), dtype=torch.float32)

    #
    image_features_by_id = {}

    for idx in tqdm(range(len(ds)), desc="Extract CLIP features"):
        item = ds.data[idx]
        image_id = int(item["image_id"])
        caption = item["text"]

        # image (dedup by image_id)
        if image_id not in image_features_by_id:
            try:
                img = Image.open(item["image_path"]).convert("RGB")
                img_in = preprocess(img).unsqueeze(0).to(device)
                img_feat = model.encode_image(img_in).float().squeeze(0).cpu()
            except Exception:
                img_feat = torch.zeros(512, dtype=torch.float32)
            image_features_by_id[image_id] = img_feat

        # text (per caption/sample)
        try:
            tokens = clip.tokenize([caption if caption else ""], truncate=True).to(device)
            txt_feat = model.encode_text(tokens).float().squeeze(0).cpu()
        except Exception:
            txt_feat = torch.zeros(512, dtype=torch.float32)

        text_features[idx] = txt_feat

    # save
    torch.save(image_features_by_id, os.path.join(cached_dir, "flickr30k_clip_image_features_by_image_id.pt"))
    torch.save(text_features, os.path.join(cached_dir, "flickr30k_clip_text_features_by_sample_idx.pt"))
    print("Saved cache to:", cached_dir)


if __name__ == "__main__":
    main()