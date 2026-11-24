import numpy as np
import json
import os
import os.path as osp
import argparse
from sklearn.model_selection import train_test_split

gt_path_tail = "gt/Train.json"

def split_json(path, split_mode):
    anno = None
    train_anno, val_anno, test_anno = {}, {}, {}
    with open(path, "r") as f:
        anno = json.load(f)

    # Assign name
    seq_name = list(anno.keys())[0]
    train_anno[seq_name] = {}
    val_anno[seq_name] = {}    
    test_anno[seq_name] = {}    

    # Get list of frame
    train_split, val_split, test_split = [int(split) for split in split_mode]
    frame_list = list(anno[seq_name].keys())
    train_frame, test_frame = train_test_split(frame_list, test_size=test_split/10.,
                                               train_size=(train_split+val_split)/10.,
                                               random_state=888, shuffle=False)
    train_frame, val_frame  = train_test_split(train_frame, test_size=val_split/(10.-test_split),
                                               train_size=train_split/(10.-test_split),
                                               random_state=888, shuffle=False)

    # 
    for frame in train_frame:
        train_anno[seq_name][frame] = anno[seq_name][frame]
    for frame in val_frame:
        val_anno[seq_name][frame] = anno[seq_name][frame]
    for frame in test_frame:
        test_anno[seq_name][frame] = anno[seq_name][frame]

    return seq_name, train_anno, val_anno, test_anno

def main():
    parser = argparse.ArgumentParser(
        prog="rtpose_annotation_split.py",
        description="For splitting data into train/val/test"
    )

    parser.add_argument(
        "-d", "--dir",
        type=str,
        help="Path to folder that keep all data"
    )

    parser.add_argument(
        "-sm", "--split_mode",
        type=str,
        default="622",
        help="Split portion for each set. For example: 622 is 0.6 train, \
              0.2 val and 0.2 test. Sum to 10."
    )

    args = parser.parse_args()
    
    seq_list = sorted(os.listdir(args.dir))
    train_set, val_set, test_set = {}, {}, {}
    for seq in seq_list:
        # Broken data
        if (seq == "civit_easy_march_02" or seq == "civit_norm_march_02"):
            continue

        # split data
        seq_name, train_anno, val_anno, test_anno = split_json(osp.join(args.dir, seq, gt_path_tail),
                                                               args.split_mode)
        
        # 
        train_set.update(train_anno)
        val_set.update(val_anno)
        test_set.update(test_anno)
    
    print(train_set.keys())

    # save
    with open(osp.join(args.dir, "../rtpose_anno/Train.json"), "w") as f:
        json.dump(train_set, f, indent=2)
    
    with open(osp.join(args.dir, "../rtpose_anno/Val.json"), "w") as f:
        json.dump(val_set, f, indent=2)
    
    with open(osp.join(args.dir, "../rtpose_anno/Test.json"), "w") as f:
        json.dump(test_set, f, indent=2)

if __name__ == "__main__":
    main()
    