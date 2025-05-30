import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

# === Register Datasets ===
dataset_root = "/home/incognito/detectron2/sam_41_mss_bl"
register_coco_instances("manuscript_train", {},
    os.path.join(dataset_root, "annotations/instances_train.json"),
    os.path.join(dataset_root, "train/images"))

register_coco_instances("manuscript_val", {},
    os.path.join(dataset_root, "annotations/instances_val.json"),
    os.path.join(dataset_root, "val/images"))

# === Config ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Core settings (unchanged from your working version)
cfg.OUTPUT_DIR = "./output_bl"
cfg.DATASETS.TRAIN = ("manuscript_train",)
cfg.DATASETS.TEST = ()  # Disable evaluation if not needed
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# === Start Training ===
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
