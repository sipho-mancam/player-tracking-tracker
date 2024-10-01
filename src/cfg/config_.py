from pathlib import Path
import os

conf_path = Path(r"./App/src/fast_reid/configs/MOT17/sbs_S50.yml").resolve()
re_id_model_path = Path(r"./App/src/fast_reid/pretrained/mot17_sbs_S50.pth").resolve()

class TrackingConf(object):
    def __init__(self) -> None:
        self.track_high_thresh = 0.8
        self.track_low_thresh = 0.55
        self.new_track_thresh = 0.5
        self.track_buffer = 30
        self.match_thresh = 0.9
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 2
        self.fuse_score = False
        self.mot20 = False
        self.cmc_method = "sparseOptFlow"
        self.name = "MOT17-01"
        self.ablation = False
        self.with_reid = False
        self.fast_reid_config = conf_path
        self.fast_reid_weights = re_id_model_path.as_posix()
        self.proximity_thresh = 0.8
        self.appearance_thresh = 0.1
        self.device = "cuda"