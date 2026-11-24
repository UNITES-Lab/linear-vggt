"""Camera-pose eval datasets (paths relative to src/, as the other eval launchers).

Standard (CUT3R / MonST3R protocol): sintel (14 seqs, full), tum (8 dynamic seqs, first 90 frames at stride 3),
scannet (100 test scenes, first 90 frames at stride 3). Long (TTT3R protocol, arXiv 2509.26645):
tum_s1_<N> (stride 1) and scannet_s3_<N> (stride 3), the first N frames; sequences shorter than N use all
their frames. Layouts come from the MonST3R / TTT3R prepare scripts.
"""
import os

SINTEL_SEQS = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2", "market_5",
               "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]

dataset_metadata = {
    "sintel": {
        "img_path": "data/eval/sintel/training/final",
        "anno_path": "data/eval/sintel/training/camdata_left",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": "sintel",
        "seq_list": SINTEL_SEQS,
    },
    "tum": {
        "img_path": "data/eval/tum",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(img_path, seq, "groundtruth_90.txt"),
        "traj_format": "tum",
        "seq_list": None,
    },
    "scannet": {
        "img_path": "data/eval/scannetv2",
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_90"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(img_path, seq, "pose_90.txt"),
        "traj_format": "replica",
        "seq_list": None,
    },
}

SCANNET_LENGTHS = [50, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                   1000]
TUM_LENGTHS = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for num in SCANNET_LENGTHS:
    dataset_metadata[f"scannet_s3_{num}"] = {
        "img_path": "data/eval/long_scannet_s3",
        "dir_path_func": lambda img_path, seq, num=num: os.path.join(img_path, seq, f"color_{num}"),
        "gt_traj_func": lambda img_path, anno_path, seq, num=num: os.path.join(img_path, seq, f"pose_{num}.txt"),
        "traj_format": "replica",
        "seq_list": None,
    }
for num in TUM_LENGTHS:
    dataset_metadata[f"tum_s1_{num}"] = {
        "img_path": "data/eval/long_tum_s1",
        "dir_path_func": lambda img_path, seq, num=num: os.path.join(img_path, seq, f"rgb_{num}"),
        "gt_traj_func": lambda img_path, anno_path, seq, num=num: os.path.join(img_path, seq,
                                                                                f"groundtruth_{num}.txt"),
        "traj_format": "tum",
        "seq_list": None,
    }
