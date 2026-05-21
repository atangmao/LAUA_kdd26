import os
import json
import argparse
import torch
import clip
from PIL import Image
from tqdm import tqdm


def resolve_image_dir(root_dir: str, split_name: str) -> str:
    """
    split_name: "train2017" or "val2017"
    Compatible with:
      root/split_name
      root/split_name/split_name   (your case)
    """
    p1 = os.path.join(root_dir, split_name)
    p2 = os.path.join(root_dir, split_name, split_name)
    return p2 if os.path.isdir(p2) else p1


def load_coco_caption_samples(ann_path: str, root_dir: str, image_dir: str):
    """
    Returns: list of samples, one caption per sample:
      {"image_id": int, "image_path": str, "text": str}
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    id2file = {int(img["id"]): img["file_name"] for img in j.get("images", [])}

    samples = []
    for ann in j.get("annotations", []):
        image_id = int(ann["image_id"])
        caption = ann.get("caption", "") or ""
        file_name = id2file.get(image_id)
        if file_name is None:
            continue

        file_name = file_name.replace("\\", "/")
        # If file_name already contains "train2017/xxx.jpg", join from root
        if "/" in file_name:
            image_path = os.path.join(root_dir, file_name)
        else:
            image_path = os.path.join(image_dir, file_name)

        samples.append({"image_id": image_id, "image_path": image_path, "text": caption})

    return samples


def get_paths(root_dir: str):
    ann_dir = os.path.join(root_dir, "annotations_trainval2017", "annotations")
    train_ann = os.path.join(ann_dir, "captions_train2017.json")
    val_ann = os.path.join(ann_dir, "captions_val2017.json")

    train_img_dir = resolve_image_dir(root_dir, "train2017")
    val_img_dir = resolve_image_dir(root_dir, "val2017")

    return ann_dir, train_ann, val_ann, train_img_dir, val_img_dir


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/home/userdata/data/MSCOCO")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--model", type=str, default="ViT-B/32")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ann_dir, train_ann, val_ann, train_img_dir, val_img_dir = get_paths(args.root_dir)

    print("root_dir   =", args.root_dir)
    print("ann_dir    =", ann_dir)
    print("train_img  =", train_img_dir)
    print("val_img    =", val_img_dir)

    if args.split in ["train", "all"]:
        assert os.path.exists(train_ann), f"Not found: {train_ann}"
    if args.split in ["val", "all"]:
        assert os.path.exists(val_ann), f"Not found: {val_ann}"

    # build samples for the requested split
    samples = []
    if args.split in ["train", "all"]:
        samples.extend(load_coco_caption_samples(train_ann, args.root_dir, train_img_dir))
    if args.split in ["val", "all"]:
        samples.extend(load_coco_caption_samples(val_ann, args.root_dir, val_img_dir))

    print(f"Loaded samples: split={args.split}, N={len(samples)}")

    model, preprocess = clip.load(args.model, device=device, jit=False)
    model.eval()

    # text features per sample_idx
    text_features = torch.empty((len(samples), 512), dtype=torch.float32)

    # image features per image_id (dedup inside this split)
    image_features_by_id = {}

    missing = 0

    for idx in tqdm(range(len(samples)), desc=f"Extract CLIP ({args.split})"):
        item = samples[idx]
        image_id = int(item["image_id"])
        caption = item["text"]
        image_path = item["image_path"]

        # image (dedup by image_id)
        if image_id not in image_features_by_id:
            try:
                if not os.path.exists(image_path):
                    missing += 1
                    raise FileNotFoundError(image_path)
                img = Image.open(image_path).convert("RGB")
                img_in = preprocess(img).unsqueeze(0).to(device)
                img_feat = model.encode_image(img_in).float().squeeze(0).cpu()
            except Exception:
                img_feat = torch.zeros(512, dtype=torch.float32)
            image_features_by_id[image_id] = img_feat

        # text (per caption/sample)
        try:
            tokens = clip.tokenize([caption], truncate=True).to(device)
            txt_feat = model.encode_text(tokens).float().squeeze(0).cpu()
        except Exception:
            txt_feat = torch.zeros(512, dtype=torch.float32)

        text_features[idx] = txt_feat

    # output filenames (split-specific)
    out_img = os.path.join(args.root_dir, f"mscoco2017_{args.split}_clip_image_features_by_image_id.pt")
    out_txt = os.path.join(args.root_dir, f"mscoco2017_{args.split}_clip_text_features_by_sample_idx.pt")

    torch.save(image_features_by_id, out_img)
    torch.save(text_features, out_txt)

    print("Saved:")
    print(" ", out_img, f"(num_images={len(image_features_by_id)})")
    print(" ", out_txt, f"(num_samples={len(text_features)})")
    print("Missing images encountered:", missing)


if __name__ == "__main__":
    main()