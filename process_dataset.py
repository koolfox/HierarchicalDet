import os
import json
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def process_coco_quadrant():
    """
    split quadrant dataset into train and val,
    copy data to coco directory
    """

    dataset_json = load_json("dentex_dataset/origin/quadrant/train_quadrant.json")

    image_ids = [x["id"] for x in dataset_json["images"]]
    random.shuffle(image_ids)
    train_ids = image_ids[: int(len(image_ids) * 0.8)]  # 80% for training

    train_json = {"images": [], "annotations": [], "categories": dataset_json["categories"]}
    val_json = {"images": [], "annotations": [], "categories": dataset_json["categories"]}

    mkdirs("dentex_dataset/coco/quadrant/train2017")
    mkdirs("dentex_dataset/coco/quadrant/val2017")

    for image in dataset_json["images"]:
        image_filename = image["file_name"]
        if image["id"] in train_ids:
            train_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant/xrays/{image_filename}",
                f"dentex_dataset/coco/quadrant/train2017/{image_filename}",
            )
        else:
            val_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant/xrays/{image_filename}",
                f"dentex_dataset/coco/quadrant/val2017/{image_filename}",
            )

    for annotation in dataset_json["annotations"]:
        if annotation["image_id"] in train_ids:
            # Make changes here to add category_id1 and set category_id2 and category_id3 to None
            annotation["category_id1"] = annotation["category_id"]
            annotation["category_id2"] = None
            annotation["category_id3"] = None
            train_json["annotations"].append(annotation)
        else:
            # Make changes here to add category_id1 and set category_id2 and category_id3 to None
            annotation["category_id1"] = annotation["category_id"]
            annotation["category_id2"] = None
            annotation["category_id3"] = None
            val_json["annotations"].append(annotation)

    mkdirs("dentex_dataset/coco/quadrant/annotations")
    save_json("dentex_dataset/coco/quadrant/annotations/instances_train2017.json", train_json)
    save_json("dentex_dataset/coco/quadrant/annotations/instances_val2017.json", val_json)

def process_coco_enumeration():
    """
    convert quadrant_enumeration label to enumeration label,
    split dataset into train and val,
    copy data to coco directory
    """

    # Load original quadrant_enumeration dataset
    dataset_json = load_json("dentex_dataset/origin/quadrant_enumeration/train_quadrant_enumeration.json")

    # Convert quadrant_enumeration labels to enumeration labels
    for annotation in dataset_json["annotations"]:
        category_id_1 = annotation["category_id_1"]
        category_id_2 = annotation["category_id_2"]

        annotation["category_id1"] = category_id_1
        annotation["category_id2"] = category_id_2
        annotation["category_id3"] = None

        annotation.pop("category_id_1")
        annotation.pop("category_id_2")

    # Shuffle and split image IDs for train (80%) and val (20%)
    image_ids = [x["id"] for x in dataset_json["images"]]
    random.shuffle(image_ids)
    train_ids = image_ids[: int(len(image_ids) * 0.8)]

    # Define categories for enumeration
    categories = [{"id": i, "name": str(i + 1), "supercategory": str(i + 1)} for i in range(32)]

    # Initialize empty dictionaries for train and val sets
    train_json = {"images": [], "annotations": [], "categories": categories}
    val_json = {"images": [], "annotations": [], "categories": categories}

    # Create directories for train and val COCO datasets
    mkdirs("dentex_dataset/coco/enumeration/train2017")
    mkdirs("dentex_dataset/coco/enumeration/val2017")

    # Copy images to train and val directories based on split
    for image in dataset_json["images"]:
        image_filename = image["file_name"]
        if image["id"] in train_ids:
            train_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_filename}",
                f"dentex_dataset/coco/enumeration/train2017/{image_filename}",
            )
        else:
            val_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration/xrays/{image_filename}",
                f"dentex_dataset/coco/enumeration/val2017/{image_filename}",
            )

    # Copy annotations to train and val sets based on split
    for annotation in dataset_json["annotations"]:
        if annotation["image_id"] in train_ids:
            train_json["annotations"].append(annotation)
        else:
            val_json["annotations"].append(annotation)

    # Create directories for annotations
    mkdirs("dentex_dataset/coco/enumeration/annotations")

    # Save train and val JSON files
    save_json("dentex_dataset/coco/enumeration/annotations/instances_train2017.json", train_json)
    save_json("dentex_dataset/coco/enumeration/annotations/instances_val2017.json", val_json)

def process_coco_disease():
    """
    extract disease label from quadrant_enumeration_disease label,
    split disease dataset into train and val,
    copy data to coco directory
    """

    dataset_json = load_json(
        "dentex_dataset/origin/quadrant_enumeration_disease/train_quadrant_enumeration_disease.json"
    )

    for annotation in dataset_json["annotations"]:
        # extract disease label from quadrant_enumeration_disease label
        category_id_1 = annotation["category_id_1"]
        category_id_2 = annotation["category_id_2"]
        category_id_3 = annotation["category_id_3"]

        annotation["category_id1"] = category_id_1
        annotation["category_id2"] = category_id_2
        annotation["category_id3"] = category_id_3

        annotation.pop("category_id_1")
        annotation.pop("category_id_2")
        annotation.pop("category_id_3")

    image_ids = [x["id"] for x in dataset_json["images"]]
    random.shuffle(image_ids)
    train_ids = image_ids[: int(len(image_ids) * 0.8)]  # 80% for training

    categories = [
        {"id": 0, "name": "Impacted", "supercategory": "Impacted"},
        {"id": 1, "name": "Caries", "supercategory": "Caries"},
        {"id": 2, "name": "Periapical Lesion", "supercategory": "Periapical Lesion"},
        {"id": 3, "name": "Deep Caries", "supercategory": "Deep Caries"},
    ]

    train_json = {"images": [], "annotations": [], "categories": categories}
    val_json = {"images": [], "annotations": [], "categories": categories}

    mkdirs("dentex_dataset/coco/disease/train2017")
    mkdirs("dentex_dataset/coco/disease/val2017")

    for image in dataset_json["images"]:
        image_filename = image["file_name"]
        if image["id"] in train_ids:
            train_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration_disease/xrays/{image_filename}",
                f"dentex_dataset/coco/disease/train2017/{image_filename}",
            )
        else:
            val_json["images"].append(image)
            shutil.copy(
                f"dentex_dataset/origin/quadrant_enumeration_disease/xrays/{image_filename}",
                f"dentex_dataset/coco/disease/val2017/{image_filename}",
            )

    for annotation in dataset_json["annotations"]:
        if annotation["image_id"] in train_ids:
            train_json["annotations"].append(annotation)
        else:
            val_json["annotations"].append(annotation)

    mkdirs("dentex_dataset/coco/disease/annotations")
    save_json("dentex_dataset/coco/disease/annotations/instances_train2017.json", train_json)
    save_json("dentex_dataset/coco/disease/annotations/instances_val2017.json", val_json)


if __name__ == "__main__":
    process_coco_quadrant()
    process_coco_enumeration()
    process_coco_disease()

    # ...
