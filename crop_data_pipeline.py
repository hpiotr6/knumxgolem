import json
import pandas as pd
import os
import cv2 as cv
from tqdm import tqdm


TARGET = "images_part1_test"  # DO EDYCJI


CROPPED_DIR = "cropped_images"
JSON_PATH = f"datas/public_evaluation/public_evaluation/{TARGET}_public.json"
IMG_DIR = f"datas/public_evaluation/public_evaluation/{TARGET}"

TARGET_DIR = os.path.join(CROPPED_DIR, TARGET)
# os.makedirs(TARGET_DIR)

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

with open(JSON_PATH) as json_data:
    data = json.load(json_data)

images_df = pd.DataFrame(data["images"])
annotations_df = pd.DataFrame(data["annotations"])
categories_df = pd.DataFrame(data["categories"])

images_annotation = pd.merge(
    annotations_df, images_df, how="inner", left_on="image_id", right_on="id"
)
assert (
    images_annotation[images_annotation["image_id"] != images_annotation["id_y"]].sum()[
        "id_x"
    ]
    == 0
)

images_annotation_categories = pd.merge(
    images_annotation, categories_df, how="inner", left_on="category_id", right_on="id"
)
images_annotation_categories = images_annotation_categories.drop(
    labels=["id_x", "id_y", "id"], axis=1
)
images_annotation_categories = images_annotation_categories.reset_index(drop=True)
images_annotation_categories = images_annotation_categories.rename(
    columns={"name": "category_name"}
)


data = {"crop_file_name": []}
for idx in tqdm(range(len(images_annotation_categories))):
    file_name = f"cropped_{idx}.png"
    data["crop_file_name"].append(file_name)

    img_row = images_annotation_categories.iloc[idx]
    img_path = os.path.join(IMG_DIR, img_row["file_name"])
    img = cv.imread(img_path)
    # plt.imshow(img);
    bbox = img_row["bbox"]
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    width = bbox[2]
    height = bbox[3]

    x2 = int(x1 + width)
    y2 = int(y1 + height)
    cropped_image = img[y1:y2, x1:x2]
    cv.imwrite(os.path.join(TARGET_DIR, file_name), cropped_image)


crop_df = pd.DataFrame(data=data)
images_annotation_categories_with_crops = pd.merge(
    images_annotation_categories,
    crop_df,
    how="inner",
    left_index=True,
    right_index=True,
)

images_annotation_categories_with_crops.to_csv(
    sep=",", path_or_buf=os.path.join(CROPPED_DIR, f"{TARGET}.csv"), index=None
)
