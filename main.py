import argparse
import copy
import os
import random
from typing import OrderedDict
import torch
import pandas as pd
import numpy as np
import cv2

from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import pydicom
import matplotlib.pyplot as plt
from accelerate.utils import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import LoggerType, ProjectConfiguration


def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:
    # https://www.kaggle.com/sadmanaraf/wheat-detection-using-faster-rcnn-train
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0
    
    ious = np.ones((len(gts), len(preds))) * -1

    for threshold in thresholds:
        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                     form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


def calculate_iou(gt, pr, form='pascal_voc') -> float:
    # https://www.kaggle.com/sadmanaraf/wheat-detection-using-faster-rcnn-train
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1
    
    if dx < 0:
        return 0.0
    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
            overlap_area
    )

    return overlap_area / union_area


def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:
    # https://www.kaggle.com/sadmanaraf/wheat-detection-using-faster-rcnn-train
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1
    for gt_idx in range(len(gts)):
        
        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue
        
        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)
            
            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:
    # https://www.kaggle.com/sadmanaraf/wheat-detection-using-faster-rcnn-train
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0
    
    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1
        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)

class RSNADataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        super().__init__()

        self.root_dir = root_dir
        self.transforms = transforms

        # load data from csv
        df = pd.read_csv(f"{root_dir}/stage_2_train_labels.csv")

        # for patient mentioned multiple times, append the bounding boxes
        patient_dict = OrderedDict()
        for idx, row in df.iterrows():
            if row['patientId'] in patient_dict:
                patient_dict[row['patientId']].append(row)
            else:
                patient_dict[row['patientId']] = [row]


        patient_data = []
        for patient_id, rows in patient_dict.items():
            patient_data.append({
                "patientId": patient_id,
                "boxes": np.array([np.array([row['x'], row['y'], row['x']+row['width'], row['y']+row['height']]) for row in rows if row['Target'] == 1]),
                "labels": np.array([1 for row in rows if row['Target'] == 1])
            })
    

        # first 90% of the data is used for training
        if split == "train":
            patient_data = patient_data[:int(0.9*len(patient_data))]
        elif split == "val":
            patient_data = patient_data[int(0.9*len(patient_data)):]
        else:
            raise ValueError("Invalid split")
        
        # manually resample train dataset so that it has equal number of samples with and without bounding boxes
        if split == "train":
            patient_data_with_boxes = [p for p in patient_data if len(p['boxes']) > 0]
            patient_data_without_boxes = [p for p in patient_data if len(p['boxes']) == 0]
            patient_data_with_boxes_resampled = random.choices(patient_data_with_boxes, k=len(patient_data_without_boxes))
            patient_data = patient_data_with_boxes_resampled + patient_data_without_boxes
            random.shuffle(patient_data)

        self.data = patient_data
        print(f"{split} dataset size: {len(self.data)}")
        
    def __getitem__(self, index: int):
        image, targets = self.raw_data(index)
        if self.transforms:
            image = self.transforms(image)
        return image, targets
    
    def raw_data(self, index: int):
        row = self.data[index]
        row = copy.deepcopy(row)
        image = pydicom.dcmread(f"{self.root_dir}/stage_2_train_images/{row['patientId']}.dcm").pixel_array
        pil_image = Image.fromarray(image).convert('RGB')
        # target class 1 indicates pneumonia is present, 0 indicates absence
        if len(row['boxes']) > 0:
            return pil_image, {'boxes': torch.tensor(row['boxes'], dtype=torch.float32), 'labels': torch.tensor(row['labels'], dtype=torch.int64)}
        else:
            return pil_image, {'boxes': torch.zeros((0, 4), dtype=torch.float32), 'labels': torch.tensor([], dtype=torch.int64)}
    
    def visualize(self, index: int, out_dir: str):
        # reset the plot
        plt.clf()
        image, targets = self.raw_data(index)
        image = np.array(image)        
        plt.imshow(image)
        for box in targets['boxes']:
            x, y, x1, y1 = box
            plt.gca().add_patch(plt.Rectangle((x, y), x1-x, y1-y, fill=False, edgecolor='r', lw=2))
        plt.savefig(f"{out_dir}/sample_{index}.png")

    def __len__(self) -> int:
        return len(self.data)

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets

def main(args):
    # make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/input_visualize", exist_ok=True)

    # setup accelerator
    config = ProjectConfiguration(
        project_dir=str(args.output_dir),
    )
    accelerator = Accelerator(
        log_with=[LoggerType.TENSORBOARD, LoggerType.COMETML],
        project_config=config,
    )
    device = accelerator.device

    # log hyperparameters
    args_dict = vars(args)
    args_dict = {k: str(v) for k, v in args_dict.items() if v is not None}
    accelerator.init_trackers("rsnadetection", args_dict)
    print("Hyperparameters: ", args_dict)

    # set seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # setup transforms using imagenet defaults
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    ])

    # setup dataset
    train_dataset = RSNADataset(args.data_dir, transforms=transform)
    valid_dataset = RSNADataset(args.data_dir, split="val", transforms=transform)
    
    # visualize few samples in the dataset
    [train_dataset.visualize(i, out_dir=f"{args.output_dir}/input_visualize") for i in range(20)]

    # setup dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # setup model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=1024, max_size=1024)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model = model.to(device)
    start_epoch = 0
    if args.resume:
        start_epoch = int(args.resume.split('_')[-1].split('.')[0])
        accelerator.print(f"Resuming from checkpoint: {args.resume}")
        model_weights = torch.load(args.resume, map_location=device)
        model.load_state_dict(model_weights)
    accelerator.print(f"No of parameters: {sum(p.numel() for p in model.parameters())}")

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # setup accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    # train model
    for epoch in range(start_epoch, args.epochs):
        accelerator.print(f"Epoch: {epoch}")
        
        # train loop
        # model.train()
        # train_loss_epoch = 0.0
        # for (images, targets) in tqdm(train_loader, f"Training Epoch {epoch}"):
        #     images = images.to(device)
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #     optimizer.zero_grad()
        #     outputs = model(images, targets)
        #     loss = sum(loss for loss in outputs.values())    
        #     train_loss_epoch += loss.item()        
        #     accelerator.backward(loss)
        #     optimizer.step()
            # import ipdb; ipdb.set_trace()

            # # total number of examples with class 1
            # total_class_1 = sum([len(t['labels']) for t in targets])
            # print(f"Total Class 1: {total_class_1}")
            
        # validate and visualize few samples
        model.eval()
        count = 0
        train_loss_epoch = 0.0
        with torch.no_grad():
            accelerator.print("Validating and Visualizing few samples")
            valid_image_precision = []
            for (images, targets) in tqdm(val_loader, f"Validating Epoch {epoch}"):
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)

                # visualize few predictions
                process_id = accelerator.process_index
                os.makedirs(f"{args.output_dir}/epoch_{epoch}", exist_ok=True)
                for i in range(len(images)):
                    image = images[i].cpu().detach().numpy().transpose(1, 2, 0)
                    image = (image * 255).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # get precision
                    boxes = outputs[i]['boxes'].detach().cpu().numpy()
                    scores = outputs[i]['scores'].detach().cpu().numpy()
                    gt_boxes = targets[i]['boxes'].detach().cpu().numpy()
                    preds_sorted_idx = np.argsort(scores)[::-1]
                    preds_sorted = boxes[preds_sorted_idx]
                    if len(gt_boxes) == 0 and len(preds_sorted) == 0:
                        image_precision = 1.0
                    else:
                        image_precision = calculate_image_precision(
                            gt_boxes,
                            preds_sorted,
                            thresholds=[0.9],
                            form='pascal_voc'
                        )
                    
                    # check if image precision is nan
                    if np.isnan(image_precision):
                        image_precision = 0.0
                        import ipdb; ipdb.set_trace()

                    valid_image_precision.append(image_precision)

                    if count > 0:
                        continue
                    for box, label, score in zip(outputs[i]['boxes'], outputs[i]['labels'], outputs[i]['scores']):
                        if label == 0 or score < 0.5:
                            continue
                        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                        cv2.putText(image, f"{score:.2f}", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imwrite(f"{args.output_dir}/epoch_{epoch}/sample_{process_id}_{i}.png", image)
                count += images.shape[0]
        
        valid_image_precision = torch.tensor(valid_image_precision).sum().item()
        print(f"Valid Image Precision: {valid_image_precision}, count: {count}")

        # accelerate sync
        t = torch.tensor([valid_image_precision, count], device=device, dtype=torch.float64)
        t_reduced = accelerator.reduce(t).cpu().numpy().tolist()
        print(f"Syn Data: {t_reduced}")
        valid_image_precision = t_reduced[0] / t_reduced[1]
                    
        # save model
        if accelerator.is_main_process:
            accelerator.log({"train_loss": train_loss_epoch, "valid_image_precision": valid_image_precision}, step=epoch)
            accelerator.print(f"Epoch: {epoch}, Train Loss: {train_loss_epoch}, Valid Image Precision: {valid_image_precision}")
            try:
                accelerator.save(model.module.state_dict(), f"{args.output_dir}/model_{epoch}.pth")
            except:
                accelerator.save(model.state_dict(), f"{args.output_dir}/model_{epoch}.pth")

        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RSNA Pneumonia Detection')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--data_dir', type=str, default='datasets/rsna-pneumonia-detection-challenge', help='data directory')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

    # parse arguments
    args = parser.parse_args()

    # update output directory with timestamp
    args.output_dir = os.path.join(args.output_dir, pd.Timestamp.now().strftime('%Y%m%d%H%M%S'))

    main(args=args)
