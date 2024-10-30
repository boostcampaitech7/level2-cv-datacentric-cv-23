import json
from typing import Dict, Any

def convert_to_UFO_format(data: Dict[str, Any]):
    UFO_format = {"images": {}}

    # imd id : 파일명 형태의 dictionary
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        image_name = image_id_to_filename[image_id]
        segmentation = annotation["segmentation"][0]

        p1 = segmentation[0:2]
        p2 = segmentation[2:4]
        p3 = segmentation[4:6]
        p4 = segmentation[6:8]

        if image_name not in UFO_format["images"]:
            UFO_format["images"][image_name] = {
                "paragraphs": {},
                "words": {},
                "chars": {},
                "img_w": data["images"][image_id - 1]["width"],
                "img_h": data["images"][image_id - 1]["height"],
                "num_patches": None,
                "tags": [], # 각자 추가하면 됩니다 ex) large_text
                "relations": {},
                "annotation_log": {
                    "worker": "worker", # 본인 이름 적으시면 되는데 귀찮으면 쓰지마세요
                    "timestamp": "2024-10-30", # 날짜인데 쓰고 싶으면 쓰세요
                    "tool_version": "CVAT",
                    "source": None
                },
                "license_tag": {
                    "usability": True,
                    "public": False,
                    "commercial": True,
                    "type": None,
                    "holder": "Upstage"
                }
            }

        UFO_format["images"][image_name]["words"][str(len(UFO_format["images"][image_name]["words"]) + 1).zfill(4)] = {
            "transcription": "",  # 비워둬야 할듯? 만약에 필요하다면 말해주세요
            "points": [p1, p2, p3, p4],
        }

    return UFO_format

# Load COCO JSON
with open("instances_Train.json") as f:
    coco_data = json.load(f)

# UFO로 변환
UFO_format_data = convert_to_UFO_format(coco_data)

# UFO JSON Save
with open("UFO_Change.json", "w") as f:
    json.dump(UFO_format_data, f, indent=4)