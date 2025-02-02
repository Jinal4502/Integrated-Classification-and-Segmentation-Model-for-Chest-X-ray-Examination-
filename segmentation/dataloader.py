import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation, SwinModel
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

class Chestxdet(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
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
        return image, mask

masks = []
images = []
for file in tqdm(os.listdir('/scratch/jjvyas1/segmentation/chestxdet/train/')):
    try:
        temp_mask = Image.open(rf'/scratch/jjvyas1/segmentation/chestxdet/train/mask/{file[:-4]}_mask.png')
        temp_image = Image.open(rf'/scratch/jjvyas1/segmentation/chestxdet/train/{file}').convert("RGB")
        images.append(temp_image)
        masks.append(temp_mask)
    except Exception as e:
        print(e, file, flush = True)

dataset = Chestxdet(images, masks)
print(dataset, flush = True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) 

print('Length of images for training: ', len(images))
print('Length of masks for training: ', len(masks))

images = []
masks = []

for file in tqdm(os.listdir('/scratch/jjvyas1/segmentation/chestxdet/test')):
    try:
        temp_mask = Image.open(rf'/scratch/jjvyas1/segmentation/chestxdet/test/mask/{file[:-4]}_mask.png')
        temp_image = Image.open(rf'/scratch/jjvyas1/segmentation/chestxdet/test/{file}').convert(    "RGB")
        images.append(temp_image)
        masks.append(temp_mask)
    except Exception as e:
        print(e, file, flush = True)

test_dataset = Chestxdet(images, masks)
print(test_dataset, flush = True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle = True)

#backbone_segmentation_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
#segmentation_config = UperNetConfig(backbone_config=backbone_segmentation_config)
#segmentation_model = UperNetForSemanticSegmentation(segmentation_config)

backbone_segmentation_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
segmentation_config = UperNetConfig(backbone_config=backbone_segmentation_config, use_pretrained_backbone=False)
model = UperNetForSemanticSegmentation(segmentation_config)
pretrained_state_dict = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").state_dict()
model.backbone.load_state_dict(SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").state_dict(), strict=False)

#config = UperNetConfig(backbone = "microsoft/swin-tiny-patch4-window7-224", use_pretrained_backbone=True)
#segmentation_model=UperNetForSemanticSegmentation(config)

device = torch.device("cuda")
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

def calculate_iou(pred_mask, true_mask):
    #print(f"Pred Mask Unique Values: {pred_mask.unique()}")
    #print(f"True Mask Unique Values: {true_mask.unique()}")
    pred_mask = pred_mask > 0.5
    true_mask = true_mask > 0
    #print(f"Pred Mask Shape: {pred_mask.shape}", f"True Mask Shape: {true_mask.shape}")
    #intersection = (pred_mask * true_mask).sum((1, 2))
    #union = (pred_mask + true_mask - pred_mask * true_mask).sum((1, 2))
    #intersection = ((pred_mask == 1) & (true_mask == 1)).sum(dim=(1, 2))
    #union = ((pred_mask == 1) | (true_mask == 1)).sum(dim=(1, 2))
    intersection = (pred_mask & true_mask).float().sum((1, 2))  # Pixel-wise AND
    union = (pred_mask | true_mask).float().sum((1, 2))
    #print('HERE')
    #intersection = np.logical_and(pred_mask, true_mask).sum()
    #union = np.logical_or(pred_mask, true_mask).sum()
    #print('HERE HERE')
    iou = intersection / (union + 1e-6)
    return iou

def precision(iou_list, threshold = 0.5):
    true_positive = (iou > threshold).sum().item()
    #true_positive = sum((iou > threshold).sum().item() for iou in iou_list)
    total = iou.numel()
    #total = len(iou_list)
    precision = true_positive / total
    return precision

for epoch in tqdm(range(300)):
    model.train()
    running_loss = 0
    iou_list = []
    #model.load_state_dict(torch.load("/scratch/jjvyas1/segmentation/chestxdet_upernet_16bs_224_1e4lr.pth"))
    for images, masks in tqdm(dataloader):
        try:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output.logits, masks.squeeze(1).long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(output.logits, dim=1)
           # preds = preds.detach().cpu().numpy()
           # masks = masks.detach().cpu().numpy()
            #print('here')
            #try:
             #   print("Unique Values in Output: ", output.logits.unique())
           # except:
            #    preds = torch.argmax(output.logits, dim=1)
             #   print("Unique Values in preds: ", preds.unique())
            #pred_probs = torch.softmax(output.logits, dim=1)
            #pred_probs_class1 = pred_probs[:, 1, :, :]
            #print("Mask Shape: ", masks.shape, flush=True)
            #print("Output Shape: ", output.logits.shape, flush=True)
            #print("Image Shape: ", images.shape, flush=True)
            iou = calculate_iou(preds, masks.squeeze(1))
           # print(iou)
            iou_list.append(iou)
        except Exception as e:
            print(e)
   # print('JINAL JINAL')
    iou_list = torch.cat(iou_list)
    mean_iou = iou_list.mean().item()
    #print('Reached till here too')
    #ap50 = precision(iou_list, threshold=0.5)
#    print("IOU List: ", iou_list, flush=True)
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}, IoU: {mean_iou}", flush=True)
    torch.save(model.state_dict(), "/scratch/jjvyas1/segmentation/5Dec_pretrained_chestxdet_upernet_16bs_224_1e4lr.pth")
    testing_iou = []
    model.eval()
    running_loss = 0
    for images, masks in tqdm(test_dataloader):
        try:
            images = images.to(device)
            masks = masks.to(device)
            #optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output.logits, masks.squeeze(1).long())
            #loss.backward()
            #optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(output.logits, dim=1)
           # preds = preds.detach().cpu().numpy()
           # masks = masks.detach().cpu().numpy()
            #print('here')
            #try:
             #   print("Unique Values in Output: ", output.logits.unique())
           # except:
            #    preds = torch.argmax(output.logits, dim=1)
             #   print("Unique Values in preds: ", preds.unique())
            #pred_probs = torch.softmax(output.logits, dim=1)
            #pred_probs_class1 = pred_probs[:, 1, :, :]
            #print("Mask Shape: ", masks.shape, flush=True)
            #print("Output Shape: ", output.logits.shape, flush=True)
            #print("Image Shape: ", images.shape, flush=True)
            iou = calculate_iou(preds, masks.squeeze(1))
           # print(iou)
            testing_iou.append(iou)
        except Exception as e:
             print(e)
    testing_iou = torch.cat(testing_iou)
    mean_iou = testing_iou.mean().item()
    print(f"Epoch {epoch+1}, Testing Loss: {running_loss/len(dataloader)}, Testing IoU: {mean_iou}", flush=True)


