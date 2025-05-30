import os
import json
import argparse
import cv2
import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from tqdm import tqdm

def setup_cfg(weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only text line class
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def create_textline_metadata():
    metadata = MetadataCatalog.get("textline_only")
    metadata.thing_classes = ["text_line"]
    metadata.thing_colors = [(0, 255, 0)]  # Green color for visualization
    return metadata

def process_image(predictor, image_path, output_dir, metadata, draw_boxes=True):
    im = cv2.imread(image_path)
    if im is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
   
    # Filter out masks smaller than a threshold area
    areas = instances.pred_masks.sum(dim=(1, 2))  # total pixels per mask
    area_threshold = 100  # ← you can increase to 200 or 300 if needed
    keep = areas > area_threshold

    # Only keep the large-enough masks
    instances = instances[keep]

    # Prepare JSON output
    annotations = {
        "image_path": os.path.basename(image_path),
        "image_size": {"height": im.shape[0], "width": im.shape[1]},
        "text_lines": []
    }
    
    # Process each detected text line
    for i in range(len(instances)):
        mask = instances.pred_masks[i].numpy()
        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append(approx.squeeze().tolist())
        
        annotations["text_lines"].append({
            "polygons": polygons,
            "confidence": float(instances.scores[i])
        })
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)
    
    # Visualization - Create a copy of instances for drawing
    from detectron2.structures import Instances
    vis_instances = Instances(instances.image_size)
    
    if draw_boxes:
        # Copy all fields for full visualization
        for field in instances._fields:
            vis_instances.set(field, instances.get(field))
    else:
        # Copy only mask-related fields
        vis_instances.set("pred_masks", instances.pred_masks)
        vis_instances.set("pred_classes", instances.pred_classes)
        vis_instances.set("scores", instances.scores)
    
    v = Visualizer(im[:, :, ::-1], 
                  metadata=metadata, 
                  scale=1.0,
                  instance_mode=ColorMode.IMAGE)
    
    out = v.draw_instance_predictions(vis_instances)
    vis_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_vis.jpg")
    cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])
    
    return annotations

def main():
    parser = argparse.ArgumentParser(description="Detect text lines in manuscript images")
    parser.add_argument("--input-dir", required=True, help="Directory containing input images")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--no-boxes", action="store_true", help="Disable bounding boxes in visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    cfg = setup_cfg(args.weights)
    predictor = DefaultPredictor(cfg)
    metadata = create_textline_metadata()
    
    supported_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    image_paths = [
        os.path.join(args.input_dir, f) 
        for f in sorted(os.listdir(args.input_dir))
        if os.path.splitext(f)[1].lower() in supported_exts
    ]
    
    print(f"Found {len(image_paths)} images to process")
    for img_path in tqdm(image_paths, desc="Processing images"):
        process_image(predictor, img_path, args.output_dir, metadata, draw_boxes=not args.no_boxes)
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    # Ignore torch.meshgrid warning - it's harmless
    import warnings
    warnings.filterwarnings("ignore", message="torch.meshgrid")
    main()
