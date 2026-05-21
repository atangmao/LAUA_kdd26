#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

def shape_of(x):
    if isinstance(x, np.ndarray):
        return x.shape
    if isinstance(x, (list, tuple)):
        return (len(x),)
    return None

def peek_one_segment(pkl_path):
    print(f"\n==== {pkl_path} ====")
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    vid = next(iter(obj.keys()))
    seg_dict = obj[vid]
    seg_id = next(iter(seg_dict.keys()))
    sample = seg_dict[seg_id]

    print("video_id:", vid)
    print("segment_id:", seg_id)
    print("segment_type:", type(sample))

    if not isinstance(sample, (tuple, list)):
        print("segment_repr:", repr(sample)[:400])
        return

    print("tuple_len:", len(sample))
    for i, elem in enumerate(sample):
        print(f"  elem_{i}: type={type(elem)}, shape={shape_of(elem)}")
        if isinstance(elem, str):
            print("     text_head:", repr(elem[:120]))
        elif isinstance(elem, np.ndarray):
            print("     array:", "shape=", elem.shape, "dtype=", elem.dtype)
        elif isinstance(elem, (int, float, np.number)):
            print("     value:", elem)

def main():
    base_dir = "/home/userdata/data/cmumosei"
    for name in ["train.pkl", "dev.pkl", "test.pkl"]:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            peek_one_segment(p)

if __name__ == "__main__":
    main()