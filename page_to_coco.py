import os
import json
import random
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse


def pagexml_to_coco_split(pagexml_dir, output_dir, split_ratio=0.8, seed=42):
    random.seed(seed)

    # Prepare directories
    train_img_dir = os.path.join(output_dir, 'train/images')
    val_img_dir = os.path.join(output_dir, 'val/images')
    ann_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    all_xmls = [f for f in os.listdir(pagexml_dir) if f.endswith('.xml')]
    random.shuffle(all_xmls)
    split_index = int(len(all_xmls) * split_ratio)
    train_files = all_xmls[:split_index]
    val_files = all_xmls[split_index:]

    def process_files(xml_files, subset_img_dir, subset_name):
        coco_output = {
            "info": {},
            "licenses": [],
            "categories": [{"id": 1, "name": "text_line"}],
            "images": [],
            "annotations": []
        }

        annotation_id = 1
        image_id = 1

        for xml_file in tqdm(xml_files, desc=f"Processing {subset_name} PAGE-XML files"):
            xml_path = os.path.join(pagexml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            page_el = root.find('.//{*}Page')
            if page_el is None:
                continue

            img_filename = page_el.attrib['imageFilename']
            img_path = os.path.join(pagexml_dir, img_filename)

            # Copy image to subset folder
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(subset_img_dir, img_filename))
            else:
                print(f"Warning: image not found {img_path}")

            width = int(page_el.attrib.get('imageWidth', 0))
            height = int(page_el.attrib.get('imageHeight', 0))

            coco_output["images"].append({
                "id": image_id,
                "file_name": img_filename,
                "width": width,
                "height": height
            })

            for textline in root.findall('.//{*}TextLine'):
                coords = textline.find('.//{*}Coords')
                if coords is None:
                    continue

                points_str = coords.attrib.get('points', '')
                polygon = []
                for point in points_str.split():
                    x, y = map(int, point.split(','))
                    polygon.extend([x, y])

                if len(polygon) < 6:
                    continue  # skip invalid polygons

                min_x = min(polygon[::2])
                max_x = max(polygon[::2])
                min_y = min(polygon[1::2])
                max_y = max(polygon[1::2])

                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [polygon],
                    "area": (max_x - min_x) * (max_y - min_y),
                    "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                    "iscrowd": 0
                })
                annotation_id += 1

            image_id += 1

        with open(os.path.join(ann_dir, f"instances_{subset_name}.json"), 'w') as f:
            json.dump(coco_output, f)

    process_files(train_files, train_img_dir, 'train')
    process_files(val_files, val_img_dir, 'val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, help='Directory with PAGE-XML files and images')
    parser.add_argument('--output-dir', required=True, help='Directory to save COCO dataset')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train split ratio (default=0.8)')
    args = parser.parse_args()

    pagexml_to_coco_split(args.input_dir, args.output_dir, args.split_ratio)

