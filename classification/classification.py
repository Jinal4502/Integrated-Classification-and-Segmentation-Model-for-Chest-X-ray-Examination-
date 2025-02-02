import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader as DataLoader, Dataset
import torch
from PIL import Image
import os
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_entry = pd.read_csv('/scratch/jjvyas1/data_entry.csv')

unique_labels = {'Effusion': 0, 'Infiltration': 1, 'Pleural_Thickening': 2, 'Pneumothorax': 3, 'Pneumonia': 4, 'Cardiomegaly': 5, 'Nodule': 6, 'Hernia': 7, 'Mass': 8, 'Edema': 9, 'Fibrosis': 10, 'Atelectasis': 11, 'Emphysema': 12, 'No Finding': 13, 'Consolidation': 14}

train_images = pd.read_csv('/scratch/jjvyas1/train_val_list.txt').iloc[:, 0].tolist()
train_images.append('00000001_000.png')

training_data = {'Image Index': [], 'Finding Labels': []}
for image in tqdm(train_images):
    try:
        training_data['Image Index'].append(data_entry[data_entry['Image Index'] == str(image)][['Image Index']].values[0][0])
        labels = data_entry[data_entry['Image Index'] == image][['Finding Labels']].values[0][0]
        temp_label = []
        for label in unique_labels:
            if label in labels:
                temp_label.append(unique_labels[f'{label}'])
        training_data['Finding Labels'].append(temp_label)
    except Exception as e:
        print(image, e)
        print(data_entry[data_entry['Image Index'] == image])
        print(image)
print(len(training_data['Image Index']), len(training_data['Finding Labels']), flush=True)
annotations = []
for image, label in tqdm(zip(training_data['Image Index'], training_data['Finding Labels'])):
    try:
       # Image.open(rf'/scratch/jjvyas1/images/{image}')
        temp_tuple = (f'{image}', label)
        annotations.append(temp_tuple)
    except Exception as e:
        print(e)

# def one_hot_encode_labels(labels, num_classes=14):
#     return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

#print(annotations[0:100])

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
        #print(labels, flush=True)
        #print(self.annotations[idx], flush=True)
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
####################################################

def calculate_auc(true_labels, predicted_probs):
    num_classes = true_labels.shape[1]
    auc_per_class = []
    
    for i in range(num_classes):
        auc = roc_auc_score(true_labels[:, i], predicted_probs[:, i])
        auc_per_class.append(auc)
    
    mean_auc = np.mean(auc_per_class)
    return auc_per_class, mean_auc


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
        temp_tuple = (f'{image}', label)
        annotations.append(temp_tuple)
    except Exception as e:
        print(e)

test_dataset = MultiLabelDataset(image_dir=rf'/scratch/jjvyas1/images/', annotations=annotations, transform=transform)

batch_size = 16
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
####################################################

model = models.convnext_base(pretrained=False)
num_classes = 14
#model.classifier[2] = torch.nn.Linear(in_features=model.classifier[2].in_features, out_features=num_classes)
model.to(device)

#model.load_state_dict(torch.load('/home/jjvyas1/convnext_weights_224_1e_5_bs16.pth'))
#print('after model')
num_epochs = 15
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()
training_losses = []
testing_losses = []
#print('before for loop')
for epoch in tqdm(range(num_epochs)):
    running_loss = 0
    model.train()
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    training_losses.append(epoch_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}', flush=True)
    testing_loss = 0
    model.eval()
    with torch.no_grad():  
        for inputs, labels in tqdm(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            testing_loss += loss.item()
    try:
        if epoch_test_loss <= testing_loss[-1]:              
            torch.save(model.state_dict(), 'convnext_weights_224_1e_5_bs16_4oct.pth')
            print('Model Saved', flush=True)
    except:
        torch.save(model.state_dict(), 'convnext_weights_224_1e_5_bs16_4oct.pth')
        print('Model Saved', flush=True)
    epoch_test_loss = testing_loss / len(test_dataloader)
    testing_losses.append(epoch_test_loss)
    print(f'Training Loss for {epoch}-epoch: ', epoch_loss, flush=True)
    print(f'Testing Loss for {epoch}-epoch: ', epoch_test_loss, flush=True)
print('Training Losses: ', training_losses, flush=True)
print('Testing Losses: ', testing_losses, flush=True)
