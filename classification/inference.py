import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader as DataLoader, Dataset
import torch
from PIL import Image
import os
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
data_entry = pd.read_csv('/scratch/jjvyas1/data_entry.csv')
 
unique_labels = {'Effusion': 0, 'Infiltration': 1, 'Pleural_Thickening': 2, 'Pneumothorax': 3, 'Pneumonia': 4, 'Cardiomegaly': 5, 'Nodule': 6, 'Hernia': 7, 'Mass': 8, 'Edema': 9, 'Fibrosis': 10, 'Atelectasis': 11, 'Emphysema': 12, 'No Finding': 13}

test_images = pd.read_csv('/scratch/jjvyas1/test_list.txt').iloc[:, 0].tolist()

testing_data = {'Image Index': [], 'Finding Labels': []}

for image in tqdm(test_images):
    try:
        testing_data['Image Index'].append(data_entry[data_entry['Image Index'] == str(image)][['Image Index']].values[0][0])
        labels = data_entry[data_entry['Image Index'] == image][['Finding Labels']].values[0][0]
        temp_label = []
        for label in unique_labels:
            if label in labels:
                temp_label.append(unique_labels[f'{label}'])
        testing_data['Finding Labels'].append(temp_label)
    except Exception as e:
        print(image, e)
        print(data_entry[data_entry['Image Index'] == image])
        print(image)
        
print(len(testing_data['Image Index']), len(testing_data['Finding Labels']), flush=True)
annotations = []
for image, label in tqdm(zip(testing_data['Image Index'], testing_data['Finding Labels'])):
    try:
       # Image.open(rf'/scratch/jjvyas1/images/{image}')
        temp_tuple = (f'{image}', label)
        annotations.append(temp_tuple)
    except Exception as e:
        print(e)

def one_hot_encode_labels(labels, num_classes=14):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class MultiLabelDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            annotations (list of tuples): List of (image_name, labels) where 
                                          labels is a tensor containing multi-label targets.
            transform (callable, optional): Optional transform to be applied
                                            on an image sample.
        """
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform
        self.num_classes = 14

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.annotations[idx][0])
        image = self.transform(Image.open(img_name).convert('RGB'))
        labels = torch.tensor(self.annotations[idx][1])
        label_vector = torch.zeros(self.num_classes)
        for l in labels:  
            label_vector[l] = 1.0
        return image, label_vector

dataset = MultiLabelDataset(image_dir=rf'/scratch/jjvyas1/images/', annotations=annotations, transform=transform)

batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#print('Completed till here!')

#try:
#    for images, labels in dataloader:
#        print(images.shape)
#        print(labels.shape)  
#except Exception as e:
#    print('here is the exception', e)

print(len(dataloader.dataset), flush=True)
print(dataloader.dataset, flush=True)

#print('before model')

model = models.convnext_base(pretrained=False)
model.to(device)

model.load_state_dict(torch.load('/home/jjvyas1/convnext_weights_224_1e_5_bs16_new.pth'))

model.eval()  
test_loss = 0
correct_predictions = 0
total_samples = 0

all_labels = []
all_probs = [] 
num_classes = 14
criterion = torch.nn.CrossEntropyLoss()

def calculate_auc(true_labels, predicted_probs):
    """
    Function to calculate AUC for each class.
    Args:
    - true_labels (numpy array): Ground truth binary labels (0 or 1) for each class.
    - predicted_probs (numpy array): Predicted probabilities for each class.

    Returns:
    - auc_per_class (list): AUC score for each class.
    - mean_auc (float): Mean AUC across all classes.
    """
    num_classes = true_labels.shape[1]
    auc_per_class = []
    
    for i in range(num_classes):
        auc = roc_auc_score(true_labels[:, i], predicted_probs[:, i])
        auc_per_class.append(auc)
    
    mean_auc = np.mean(auc_per_class)
    return auc_per_class, mean_auc


with torch.no_grad():  
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        #print(outputs, flush=True)
        predicted_probs = torch.sigmoid(outputs)
        all_probs.append(predicted_probs.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_probs).numpy()
all_labels = torch.cat(all_labels).numpy()
#print(all_preds, all_labels)
# print(set(all_labels), flush=True)

print('here', flush=True)

auc_per_class, mean_auc = calculate_auc(all_labels, all_preds)

print('here', flush=True)

for idx, auc in enumerate(auc_per_class):
    print(f"AUC for class {idx + 1}: {auc:.4f}", flush=True)
print(f"Mean AUC across all classes: {mean_auc:.4f}", flush=True)
