from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from gfpgan import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
import torch
import head_segmentation.segmentation_pipeline as seg_pipeline


def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 0.0, bias: float = 50.0
) -> List[Image.Image]:
    """
    Returns a list of images with masks on the face parts.
    """
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.1
    )
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.1
    )

    masks = []
    for image in tqdm(images):
        image_np = np.array(image)

        # Perform face detection
        results_detection = face_detection.process(image_np)
        ih, iw, _ = image_np.shape
        if results_detection.detections:
            for detection in results_detection.detections:
                bboxC = detection.location_data.relative_bounding_box

                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # make sure bbox is within image
                bbox = (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(iw - bbox[0], bbox[2]),
                    min(ih - bbox[1], bbox[3]),
                )

                print(bbox)

                # Extract face landmarks
                face_landmarks = face_mesh.process(
                    image_np[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                ).multi_face_landmarks

                # https://github.com/google/mediapipe/issues/1615
                # This was def helpful
                indexes = [
                    10,
                    338,
                    297,
                    332,
                    284,
                    251,
                    389,
                    356,
                    454,
                    323,
                    361,
                    288,
                    397,
                    365,
                    379,
                    378,
                    400,
                    377,
                    152,
                    148,
                    176,
                    149,
                    150,
                    136,
                    172,
                    58,
                    132,
                    93,
                    234,
                    127,
                    162,
                    21,
                    54,
                    103,
                    67,
                    109,
                ]

                if face_landmarks:
                    mask = Image.new("L", (iw, ih), 0)
                    mask_np = np.array(mask)

                    for face_landmark in face_landmarks:
                        face_landmark = [face_landmark.landmark[idx] for idx in indexes]
                        landmark_points = [
                            (int(l.x * bbox[2]) + bbox[0], int(l.y * bbox[3]) + bbox[1])
                            for l in face_landmark
                        ]
                        mask_np = cv2.fillPoly(
                            mask_np, [np.array(landmark_points)], 255
                        )

                    mask = Image.fromarray(mask_np)

                    # Apply blur to the mask
                    if blur_amount > 0:
                        mask = mask.filter(ImageFilter.GaussianBlur(blur_amount))

                    # Apply bias to the mask
                    if bias > 0:
                        mask = np.array(mask)
                        mask = mask + bias * np.ones(mask.shape, dtype=mask.dtype)
                        mask = np.clip(mask, 0, 255)
                        mask = Image.fromarray(mask)

                    # Convert mask to 'L' mode (grayscale) before saving
                    mask = mask.convert("L")

                    masks.append(mask)
                else:
                    # If face landmarks are not available, add a black mask of the same size as the image
                    masks.append(Image.new("L", (iw, ih), 255))

        else:
            print("No face detected, adding full mask")
            # If no face is detected, add a white mask of the same size as the image
            masks.append(Image.new("L", (iw, ih), 255))

    return masks


def _center_of_mass_and_bounding_box(mask: Image.Image, threshold: float = 0.6):
    """
    Returns the center of mass of the mask and the width and height of the bounding box
    that considers only white areas above the specified threshold (default is 60% white).
    """
    # Convert image to numpy array and apply threshold
    mask_np = np.array(mask)
    mask_thresholded = np.where(mask_np > (threshold * 255), 255, 0)

    # Center of mass calculation
    x, y = np.meshgrid(np.arange(mask_np.shape[1]), np.arange(mask_np.shape[0]))
    total = np.sum(mask_thresholded)

    if total == 0:
        return 0, 0, 0, 0
    x_com = np.sum(x * mask_thresholded) / total
    y_com = np.sum(y * mask_thresholded) / total

    # Bounding box calculation for white areas above the threshold
    white_pixels = np.where(mask_thresholded == 255)
    if white_pixels[0].size == 0 or white_pixels[1].size == 0:  # No white pixels found
        return int(x_com), int(y_com), 0, 0

    x_min, x_max = np.min(white_pixels[1]), np.max(white_pixels[1])
    y_min, y_max = np.min(white_pixels[0]), np.max(white_pixels[0])

    # The width and height of the bounding box without padding
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    return int(x_com), int(y_com), bbox_width, bbox_height


# Example usage:
# Load your image as a PIL Image object and call the function
# x_com, y_com, width, height, masked_part = _center_of_mass_and_bounding_box(your_image)


def _crop_to_square_and_bounding_box(
    image: Image.Image,
    com: Tuple[int, int],
    bbox_dims: Tuple[int, int],
    resize_to: Optional[int] = None,
    padding: int = 0,
):
    cx, cy = com
    bbox_width, bbox_height = bbox_dims
    width, height = image.size

    # Use the larger of bbox_width and bbox_height for square crop dimensions
    # Add padding to the square crop area
    side_length = max(bbox_width, bbox_height) + padding

    # Adjust center of mass if too close to the edges
    cx = max(side_length // 2, min(cx, width - side_length // 2))
    cy = max(side_length // 2, min(cy, height - side_length // 2))

    # Determine crop dimensions based on the square crop area
    left = cx - side_length // 2
    right = cx + side_length // 2
    top = cy - side_length // 2
    bottom = cy + side_length // 2

    # Crop the image
    image = image.crop((left, top, right, bottom))

    print("IMAGE SIZE CROPPED", image.size)

    # Resize if required
    if resize_to:
        image = image.resize((resize_to, resize_to), Image.Resampling.LANCZOS)

    # Third arg is width and height of bounding box
    return image, (left, top), (right - left, bottom - top)


def crop_faces_to_square(
    original_image: Image.Image,
    mask_image: Image.Image,
    control_image: Optional[Image.Image] = None,
    padding: Optional[float] = 0.5,
    resize_to: Optional[int] = 768,
) -> Tuple[
    Image.Image, Image.Image, Optional[Image.Image], Tuple[int, int], Tuple[int, int]
]:
    # Find the center of mass of the mask
    com = _center_of_mass_and_bounding_box(mask_image)
    print(com)
    padding = int(min(com[2], com[3]) * padding)

    # Based on the center of mass, crop the image to a square
    image, left_top, crop_size = _crop_to_square_and_bounding_box(
        original_image,
        [com[0], com[1]],
        [com[2], com[3]],
        resize_to=resize_to,
        padding=padding,
    )
    mask, _, _ = _crop_to_square_and_bounding_box(
        mask_image,
        [com[0], com[1]],
        [com[2], com[3]],
        resize_to=resize_to,
        padding=padding,
    )

    control = None
    if control_image:
        control, _, _ = _crop_to_square_and_bounding_box(
            control_image,
            [com[0], com[1]],
            [com[2], com[3]],
            resize_to=resize_to,
            padding=padding,
        )

    print("left top 2", left_top)
    return image, mask, control, left_top, crop_size


def paste_inpaint_into_original_image(
    original_image: Image.Image,
    left_top: Tuple[int, int],
    image_to_paste: Image.Image,
    paste_size: Tuple[int, int],
) -> Image.Image:
    """
    Paste an image back into its original position in the larger image.

    :param original_image: The original larger image.
    :param left_top: The (x, y) coordinates of the top left corner where to paste the image.
    :param image_to_paste: The image to paste into the original image.
    :return: The final merged image.
    """
    # Resize the image to be pasted to the specified paste size
    image_to_paste = image_to_paste.resize(paste_size, Image.Resampling.LANCZOS)

    # Create a copy of the original image to avoid modifying it directly
    final_image = original_image.copy()

    # Paste the new image into the original image at the specified coordinates and using the mask
    final_image.paste(image_to_paste, left_top)

    return final_image


def get_head_mask(
    head_image: Image.Image,
    blur_amount: float = 8.0,
) -> Image.Image:
    # Setting up the device for the segmentation pipeline
    device = torch.device("cuda")

    # Initialize the segmentation pipeline
    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=device)

    # Convert the image to a numpy array
    image_np = np.array(head_image)

    # Predict the segmentation map
    segmentation_map = segmentation_pipeline.predict(image_np)

    # Create a mask where the head is white and the rest is black
    head_mask = np.where(segmentation_map == 1, 255, 0).astype(np.uint8)

    # Convert the mask to a 3-channel image (white head, black background)
    segmented_region = cv2.cvtColor(head_mask, cv2.COLOR_GRAY2RGB)

    # Convert the result to a PIL image
    pil_image = Image.fromarray(segmented_region)

    # Apply blur amount
    pil_image_blur = pil_image.filter(ImageFilter.GaussianBlur(blur_amount))
    return pil_image_blur
