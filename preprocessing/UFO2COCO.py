import json
from typing import Dict, Any

def convert_to_coco_format(data: Dict[str, Any]):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "ocr"}],
    }

    image_id_counter = 1
    annotation_id_counter = 1

    for file_name, file_data in data["images"].items():
        image_id = image_id_counter
        image_id_counter += 1

        coco_image = {
            "id": image_id,
            "width": file_data["img_w"],
            "height": file_data["img_h"],
            "file_name": file_name,
            "license": 0, 
            "flickr_url": None, 
            "coco_url": None, 
            "date_captured": 0  
        }
        coco_data["images"].append(coco_image)

        for word_id, word_data in file_data["words"].items():
            annotation_id = annotation_id_counter
            annotation_id_counter += 1

            [p1, p2, p3, p4] = word_data["points"]

            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [[p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]]],
                "area": 0,
                "bbox": [],
                "iscrowd": 0  
            }
            coco_data["annotations"].append(coco_annotation)

    return coco_data


# Load UFO json
with open("train.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to COCO
coco_data = convert_to_coco_format(data)

# Save COCO json
with open("train_coco.json", "w") as f:
    json.dump(coco_data, f, indent=4)