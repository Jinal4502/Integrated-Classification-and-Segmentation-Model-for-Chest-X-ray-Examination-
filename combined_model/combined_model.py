import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
from torchvision.models import convnext_base
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation, SwinModel
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import json
from torch import nn

class Chestxdet(Dataset):
    def __init__(self, images, masks, labels):
        self.images = images
        self.masks = masks
        self.labels = labels
        self.num_classes = 14
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.transform(self.images[index])
        mask = self.transform(self.masks[index])
        mask = np.array(mask)
        mask[mask > 0] = 1
        label = torch.tensor(self.labels[index])
        labels = torch.zeros(self.num_classes)
        for l in label:
            labels[l] = 1.0
        return image, mask, labels

with open("/scratch/jjvyas1/segmentation/dict_labels.txt", "r") as file:
    dict_labels = json.load(file)

masks = []
images = []
labels = []
unique_labels = {'Effusion': 0, 'Calcification': 1, 'Pleural_Thickening': 2, 'Pneumothorax': 3, 'Nodule': 4, 'Cardiomegaly': 5, 'Diffuse Nodule': 6, 'Fracture': 7, 'Mass': 8, 'Fibrosis': 9, 'Atelectasis': 10, 'Emphysema': 11, 'Consolidation': 12}
for file in tqdm(os.listdir('/scratch/jjvyas1/segmentation/chestxdet/train/')):
    try:
        temp_mask = Image.open(rf'/scratch/jjvyas1/segmentation/chestxdet/train/mask/{file[:-4]}_mask.png')
        temp_image = Image.open(rf'/scratch/jjvyas1/segmentation/chestxdet/train/{file}').convert("RGB")
        images.append(temp_image)
        masks.append(temp_mask)
        try:
            labels_str = dict_labels[file]
            temp_labels = []
            for label in unique_labels:
                if label in labels_str:
                    temp_labels.append(unique_labels[f'{label}'])
            labels.append(temp_labels)
        except Exception as e:
            print(f'Could not retrieve labels for {file}, with error, {e}')
            labels.append([13]) #13 as No Finding Label
        #print(labels_str, labels, flush=True)
    except Exception as e:
        print(e, file, flush = True)
#print(labels)

dataset = Chestxdet(images, masks, labels)
#print(dataset, flush = True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) 

print('Length of images for training: ', len(images))
print('Length of masks for training: ', len(masks))
print('Length of labels for training: ', len(labels))

#images = []
#masks = []

#for file in tqdm(os.listdir('/scratch/jjvyas1/segmentation/chestxdet/test')):
#    try:
#        temp_mask = Image.open(rf'/scratch/jjvyas1/segmentation/chestxdet/test/mask/{file[:-4]}_mask.png')
#        temp_image = Image.open(rf'/scratch/jjvyas1/segmentation/chestxdet/test/{file}').convert(    "RGB")
#        images.append(temp_image)
#        masks.append(temp_mask)
#    except Exception as e:
#        print(e, file, flush = True)

#test_dataset = Chestxdet(images, masks)
#print(test_dataset, flush = True)
#test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle = True)


backbone_segmentation_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
segmentation_config = UperNetConfig(backbone_config=backbone_segmentation_config, use_pretrained_backbone=False)
segmentation_model = UperNetForSemanticSegmentation(segmentation_config)
pretrained_state_dict = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").state_dict()
segmentation_model.backbone.load_state_dict(SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").state_dict(), strict=False)

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
      super(CombinedModel, self).__init__()
      self.segmentation_model = segmentation_model
      self.classification_head = convnext_base(pretrained=True)
      self.classification_head.classifier[2] = torch.nn.Linear(in_features=self.classification_head.classifier[2].in_features, out_features=num_classes)
      self.channel_converter = nn.Conv2d(2, 3, kernel_size=1)
    def forward(self, x):
      seg_outputs = self.segmentation_model(x)
      #print('here')
      pooled_output = self.channel_converter(seg_outputs['logits'])  # Adjust based on features
      # print(seg_outputs)
      # pooled_output = pooled_output.mean(dim=[2, 3])  # Average pooling
      #print(x.shape)
      #print(pooled_output.shape)
      # print(x.shape)
      class_logits = self.classification_head(pooled_output)
      return seg_outputs, class_logits

def calculate_iou(pred_mask, true_mask):
    pred_mask = pred_mask > 0.5
    true_mask = true_mask > 0
    intersection = (pred_mask & true_mask).float().sum((1, 2))  # Pixel-wise AND
    union = (pred_mask | true_mask).float().sum((1, 2))
    iou = intersection / (union + 1e-6)
    return iou


def train(model, dataloader, optimizer, segmentation_loss_fn, classification_loss_fn, device):
    model.train()
    losses = []
    ious = []
    for images, masks, labels in dataloader:
        try:
            images = images.to(device)
            masks = masks.to(device)
            #print(images.shape, masks.shape, 'here', flush=True)
            labels = labels.to(device)
            optimizer.zero_grad()
            #print('here')
            seg_outputs, class_logits = model(images)
            #print('here again')
            seg_loss = segmentation_loss_fn(seg_outputs['logits'], masks.squeeze(1).long())
            class_loss = classification_loss_fn(class_logits, labels.type(torch.float32))
            #print('here again again')
            total_loss = seg_loss + class_loss
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss)
            seg_preds = torch.argmax(seg_outputs['logits'], dim=1)
            ious.append(calculate_iou(seg_preds, masks.squeeze(1)))
        except Exception as e:
            print(e, flush=True)
            print(f'Could not train for batch with this image and mask shape, {images.shape}, {masks.shape}')
        return losses, torch.cat(ious).mean().item()

# Model, Optimizer, and Loss Functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedModel(num_classes=14).to(device)  # Update classes accordingly
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
segmentation_loss_fn = nn.functional.cross_entropy
classification_loss_fn = nn.BCEWithLogitsLoss()

# Training
epoch_losses = []
epoch_ious = []
for i in range(5):
  loss, iou = train(model, dataloader, optimizer, segmentation_loss_fn, classification_loss_fn, device)
  epoch_losses.append(loss)
  epoch_ious.append(iou)
print('Losses: ', epoch_losses)
print('IoUs: ', epoch_ious) 
#print(train(model, dataloader, optimizer, segmentation_loss_fn, classification_loss_fn, device))
