import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
from custom_layers.roi_heads import CosineROIHeads, WeightedEntropyROIHeads

setup_logger()
trn_json = "VOS_DATASET_ROOT/voc0712_train_all.json"
val_json = "VOS_DATASET_ROOT/val_coco_format.json"
img_dir = "/home/wjdgy/SAM-OOD-Detection/VOS_DATASET_ROOT/JPEGImages"
register_coco_instances("train_dataset", {}, trn_json, img_dir)
register_coco_instances("val_dataset", {}, val_json, img_dir)


cfg = get_cfg()
cfg.merge_from_file("configs/PascalVOC-Detection/scaled_cosine/scaled_cosine.yaml")
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.VAL = ("val_dataset",)
cfg.OUTPUT_DIR = 'scaled_cosine'
cfg.DATASETS.TEST = ()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20 + 1  # 20 known classes + 1 unknown class
if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
