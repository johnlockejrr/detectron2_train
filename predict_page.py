import os
import json
import argparse
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
import numpy as np
from scipy.spatial import ConvexHull

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

def calculate_baseline(polygon):
    """Calculate baseline that follows the polygon shape.

    Args:
        polygon: List of (x,y) points representing the text line polygon

    Returns:
        List of [[x1,y1], [x2,y2]] points representing the baseline
    """
    if len(polygon) < 2:
        return []

    poly = np.array(polygon)

    # Calculate middle height (median of y-coordinates)
    y_coords = poly[:,1]
    middle_y = np.median(y_coords)

    # Find convex hull for ordered points
    try:
        hull = ConvexHull(poly)
        ordered_points = poly[hull.vertices]
    except:
        # Fallback if convex hull fails (e.g., colinear points)
        ordered_points = poly

    # Find intersections with middle_y line
    intersections = []
    n = len(ordered_points)

    for i in range(n):
        p1 = ordered_points[i]
        p2 = ordered_points[(i+1)%n]

        # Check if segment crosses middle_y
        if (p1[1] >= middle_y and p2[1] <= middle_y) or (p1[1] <= middle_y and p2[1] >= middle_y):
            if p2[0] == p1[0]:  # Vertical line
                x = p1[0]
            else:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                x = p1[0] + (middle_y - p1[1]) / slope
            intersections.append([x, middle_y])

    # If we found at least 2 intersections, use leftmost and rightmost
    if len(intersections) >= 2:
        intersections.sort()
        left_point = intersections[0]
        right_point = intersections[-1]

        # Extend slightly beyond polygon bounds (10px)
        left_point[0] = max(0, left_point[0] - 10)  # Don't go below 0
        right_point[0] = min(img_width, right_point[0] + 10)  # Don't exceed image width

        return [left_point, right_point]

    # Fallback: use min/max x with middle y
    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    return [[max(0, x_min - 10), middle_y],
            [min(img_width, x_max + 10), middle_y]]

def create_page_xml(image_path, polygons, output_dir):
    """Create PAGE-XML file with proper polygon-shaped baselines"""
    global img_width  # Used in baseline calculation
    img_filename = os.path.basename(image_path)
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Create XML structure
    pcgts = ET.Element("PcGts", {
        "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"
    })
    
    # Metadata
    metadata = ET.SubElement(pcgts, "Metadata")
    ET.SubElement(metadata, "Creator").text = "Detectron2"
    now = datetime.now().isoformat()
    ET.SubElement(metadata, "Created").text = now
    ET.SubElement(metadata, "LastChange").text = now
    
    # Page element
    page = ET.SubElement(pcgts, "Page", {
        "imageFilename": img_filename,
        "imageWidth": str(img_width),
        "imageHeight": str(img_height)
    })
    
    if not polygons:
        return None
    
    # Create parent TextRegion (bounding box of all text lines + padding)
    all_points = [point for polygon in polygons for point in polygon]
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    
    min_x = max(0, min(x_coords) - 20)
    max_x = min(img_width, max(x_coords) + 20)
    min_y = max(0, min(y_coords) - 20)
    max_y = min(img_height, max(y_coords) + 20)
    
    region_coords = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y)
    ]
    region_coords_str = " ".join(f"{x},{y}" for x, y in region_coords)
    
    text_region = ET.SubElement(page, "TextRegion", {
        "id": "region_1",
        "custom": "structure {type:text_zone;}"
    })
    ET.SubElement(text_region, "Coords", {"points": region_coords_str})
    
    # Add all text lines with polygon-shaped baselines
    for i, polygon in enumerate(polygons):
        text_line = ET.SubElement(text_region, "TextLine", {
            "id": f"line_{i+1}",
            "custom": "structure {type:text_line;}"
        })
        
        # Convert polygon to PAGE-XML coordinate string
        line_coords = " ".join(f"{int(x)},{int(y)}" for x, y in polygon)
        ET.SubElement(text_line, "Coords", {"points": line_coords})
        
        # Calculate baseline following polygon shape
        baseline_points = calculate_baseline(polygon)
        if baseline_points:
            baseline_str = " ".join(f"{int(x)},{int(y)}" for x, y in baseline_points)
            ET.SubElement(text_line, "Baseline", {"points": baseline_str})
        
        # Empty TextEquiv
        text_equiv = ET.SubElement(text_line, "TextEquiv", {"conf": "1.0"})
        ET.SubElement(text_equiv, "Unicode")
    
    # Save XML
    xml_str = ET.tostring(pcgts, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    xml_filename = f"{os.path.splitext(img_filename)[0]}.xml"
    xml_path = os.path.join(output_dir, xml_filename)
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    return xml_path

def process_image(predictor, image_path, output_dir, metadata, draw_boxes=True):
    im = cv2.imread(image_path)
    if im is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    
    # Prepare JSON output
    annotations = {
        "image_path": os.path.basename(image_path),
        "image_size": {"height": im.shape[0], "width": im.shape[1]},
        "text_lines": []
    }
    
    # Process polygons for PAGE-XML
    polygons_for_xml = []
    
    # Process each detected text line
    for i in range(len(instances)):
        mask = instances.pred_masks[i].numpy()
        contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append(approx.squeeze().tolist())
        
        # For JSON output
        annotations["text_lines"].append({
            "polygons": polygons,
            "confidence": float(instances.scores[i])
        })
        
        # For PAGE-XML (flattened list of points)
        polygons_for_xml.extend([[(int(p[0]), int(p[1])) for p in poly] for poly in polygons])
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.json")
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2)
    
    # Save PAGE-XML
    if polygons_for_xml:
        xml_path = create_page_xml(image_path, polygons_for_xml, output_dir)
    
    # Visualization
    v = Visualizer(im[:, :, ::-1], 
                  metadata=metadata, 
                  scale=1.0,
                  instance_mode=ColorMode.IMAGE)
    
    if not draw_boxes:
        # Create new instances without boxes
        from detectron2.structures import Instances
        vis_instances = Instances(instances.image_size)
        vis_instances.set("pred_masks", instances.pred_masks)
        vis_instances.set("pred_classes", instances.pred_classes)
        vis_instances.set("scores", instances.scores)
        instances = vis_instances
    
    out = v.draw_instance_predictions(instances)
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
