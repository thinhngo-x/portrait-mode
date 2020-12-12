"""Main script."""

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import cv2
import torchvision
from matplotlib import pyplot as plt
import argparse


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--img_path', type=str, default=os.path.join('sample_images', '154.jpg'))
    parser.add_argument('--save_img', type=bool, default=False)
    parser.add_argument('--gpu', type=bool, default=False)
    args = parser.parse_args()
    return args


def load_model(download=True):
    """Load pre-trained Mask R-CNN with ResNet50 as backbone."""
    print("Loading model...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=download,
        pretrained_backbone=download
    )
    if not download:
        model.load_state_dict(torch.load(os.path.join('model', 'maskrcnn_resnet50_fpn.ptn')))
    else:
        torch.save(model.state_dict(), os.path.join('model', 'maskrcnn_resnet50_fpn.ptn'))
    print("Done!")
    return model


def IoU(prediction, id1, id2, device=torch.device('cpu')):
    """Compute IoU of two bounding boxes at index id1 and id2 of the prediction."""
    # First extract the bounding boxes for each mask.
    bbox1 = prediction['boxes'][id1]
    bbox2 = prediction['boxes'][id2]
    device = bbox1.device

    # Compute U.
    xmin = min(bbox1[0], bbox2[0])
    ymin = min(bbox1[1], bbox2[1])
    xmax = max(bbox1[2], bbox2[2])
    ymax = max(bbox1[3], bbox2[3])
    u = (xmax - xmin) * (ymax - ymin)
    union_bbox = torch.FloatTensor([xmin, ymin, xmax, ymax]).to(device)

    # Compute I.
    xmin = max(bbox1[0], bbox2[0])
    ymin = max(bbox1[1], bbox2[1])
    xmax = min(bbox1[2], bbox2[2])
    ymax = min(bbox1[3], bbox2[3])
    i = (xmax - xmin) * (ymax - ymin)
    inter_bbox = torch.FloatTensor([xmin, ymin, xmax, ymax]).to(device)

    return i / u, inter_bbox, union_bbox


def IoU_mask(prediction, id1, id2, thres):
    """Compute IoU  of 2 masks."""
    m1 = prediction['masks'][id1, 0]
    m2 = prediction['masks'][id2, 0]
    inter = (m1 > thres) & (m2 > thres)
    union = (m1 > thres) | (m2 > thres)

    return torch.sum(inter).float() / torch.sum(union)


def merge_masks(m1, m2):
    """Merge 2 masks."""
    return torch.max(m1, m2)


def get_mask(prediction, thres_merge_per=0.1, thres_merge_obj=0.2, thres=0.1):
    """Merge or select masks for the foreground objects and persons."""
    num_objs = prediction['labels'].shape[0]
    # print(prediction['labels'] == 1)

    # Merge mask of crowds
    idx_person = torch.arange(num_objs)[prediction['labels'] == 1]
    # print(idx_person)
    cur_idx = idx_person[0]
    cur_area = torch.sum(prediction['masks'][cur_idx, 0] > thres)

    for i in range(1, len(idx_person)):
        iou, inter, union = IoU(prediction, cur_idx, idx_person[i])
        # print(iou)
        if iou > thres_merge_per:
            # Then merge two masks
            prediction['masks'][cur_idx, 0] = merge_masks(prediction['masks'][cur_idx, 0],
                                                          prediction['masks'][idx_person[i], 0])
            prediction['boxes'][cur_idx, :] = union
            cur_area = torch.sum(prediction['masks'][cur_idx, 0] > thres)
        else:
            # Compare areas between masks and pick the mask with larger area
            area_i = torch.sum(prediction['masks'][idx_person[i], 0] > 0)
            if area_i > cur_area:
                cur_area = area_i
                cur_idx = i

    # Merge masks of objects overlapping on persons
    idx_obj = torch.arange(num_objs)[prediction['labels'] != 1]
    for i in range(len(idx_obj)):
        # If the object lies inside the mask of person, merge them
        iou = IoU_mask(prediction, cur_idx, idx_obj[i], thres)
        if iou > thres_merge_obj:
            prediction['masks'][cur_idx, 0] = merge_masks(prediction['masks'][cur_idx, 0],
                                                          prediction['masks'][idx_obj[i], 0])

    return prediction['masks'][cur_idx, 0]


def apply_blur(image, prediction, thres):
    """Synthesize image with bokeh effect."""
    h, w = image.shape[0:2]
    ksize = min(h, w) // 20
    if ksize % 2 == 0:
        ksize += 1

    mask = get_mask(prediction, thres=thres).detach().cpu().numpy()
    mask[mask > thres] = 1.0
    mask[mask < thres] = 0.0
    mask = cv2.erode(mask, np.ones((ksize // 2, ksize // 2), dtype=np.uint8))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((ksize, ksize), dtype=np.uint8))
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    # print(np.unique(mask))
    mask = np.expand_dims(mask, 2)
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    # print(mask)
    image = image * mask + blurred * (1 - mask)
    return image.astype(np.uint8)


if __name__ == "__main__":
    args = parse_args()
    args = vars(args)

    if args['gpu']:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("Warning: CUDA is not available, switching to CPU!")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    if len(os.listdir('model')) == 0:  # Model state dict is not available
        # Then download it
        mask_model = load_model(download=True).to(device)
    else:
        mask_model = load_model(download=False).to(device)

    image = Image.open(args['img_path'])
    plt.imshow(np.array(image))
    img_tensor = torchvision.transforms.functional.to_tensor(image)

    # Predict masks
    mask_model.eval()
    with torch.no_grad():
        predictions = mask_model([img_tensor.to(device)])

    out = apply_blur(np.array(image), predictions[0], thres=0.5)
    cv2.imshow("Output", out)
