import torch.nn as nn
import torch.optim as optim
import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import spacy
import pandas as pd
from torch.utils.data import Subset
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import random
import time

# J'ai fait tourné sur 2 pc différents, un qui avait une gpu, l'autre non
print("CUDA Available:", torch.cuda.is_available())

# Nombre de Gpus
print("Number of GPUs:", torch.cuda.device_count())
print(torch.version.cuda)

#Version de torch pour pouvoir installé cuda 
print(torch.__version__)


# Nom du Gpu utilisé
#print("Current GPU:", torch.cuda.get_device_name(0))

spacy_tokenizer = spacy.load("en_core_web_sm")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#class Vocabulary
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }
        self.stoi = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3
        }
        self.freq_threshold = freq_threshold

    def tokenize(self, text):
        return [token.text.lower() for token in spacy_tokenizer.tokenizer(text)]

    def make_vocabulary(self, sequences):
        current_idx = 4
        frequencies = {}

        for sequence in sequences:
            for word in self.tokenize(sequence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = current_idx
                    self.itos[current_idx] = word
                    current_idx += 1

    def encode(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]

    def decode(self, sequence):
        return [self.itos[token] if token in self.itos else "<UNK>" for token in sequence]

    def __len__(self):
        return len(self.itos)

# Classe CustomPKMNDataset
class CustomPKMNDataset(Dataset):
    def __init__(self, img_dir, data_file, transform=None):
        self.img_dir = img_dir
        self.data = pd.read_csv(data_file)
        self.max_length = self.data['caption'].apply(lambda x: len(x.split())).max()
        self.transform = transform
        self.vocab = Vocabulary()
        self.vocab.make_vocabulary(self.data['caption'])
        self.uniques = self.data['type'].unique()
        self.t_to_i = {type: idx for idx, type in enumerate(self.uniques)}
        self.i_to_t = {idx: type for type, idx in self.t_to_i.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ligne = self.data.iloc[idx]
        image_name = ligne['image']

        img_path = os.path.join(self.img_dir, image_name)

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        caption = ligne['caption']
        type = ligne['type']

        legende = [1] + self.vocab.encode(caption)[1:] + [2]
        add_padding = self.max_length + 2 - len(legende)
        legende += [0] * add_padding

        type_encodage = self.t_to_i[type]

        return image, torch.tensor(legende, dtype=torch.long), torch.tensor(type_encodage, dtype=torch.long)

# Classe de PaddingCollate
class PaddingCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        types = [item[2].unsqueeze(0) for item in batch]
        types = torch.cat(types, dim=0)
        return imgs, targets, types

# Fonction pour créer les loaders
# On drop last ici pour eviter d'avoir un batch non complet a la fin (j'ai eu des bugs à la fin des batchs durant les entrainements)
def make_loader(img_dir, data_file, transform, batch_size=32, num_workers=0, shuffle=True, pin_memory=True):
    dataset = CustomPKMNDataset(img_dir, data_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    #on prend 80-20 comme repartition
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
        pin_memory=pin_memory, collate_fn=PaddingCollate(pad_idx), drop_last=True)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
        pin_memory=pin_memory, collate_fn=PaddingCollate(pad_idx), drop_last=True)

    train_loader_complete = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
        pin_memory=pin_memory, collate_fn=PaddingCollate(pad_idx), drop_last=True)

    return train_loader_complete, (train_loader, test_loader), dataset

# Classe InfiniteDataLoader

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=0, shuffle=True, pin_memory=True):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        pad_idx = dataset.vocab.stoi["<PAD>"]

        self.train_loader = DataLoader(
            dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,
            pin_memory=pin_memory, collate_fn=PaddingCollate(pad_idx))
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

# Chemins des données
img_dir = 'data/images'
data_file = 'data/data.csv'

# Taille minimale des images et taille d'entrée pour ResNet
# On checke la taille des images dans notre set
set_size = set()

def taille_img(img_dir):
    for name in os.listdir(img_dir):
        path = os.path.join(img_dir, name)
        with Image.open(path) as img:
            set_size.add(img.size[0])

taille_img(img_dir)

size_resnet = 224
min_size = min(set_size)

# Affichage des tailles
print(f"Taille minimale {min_size}, taille d'entrée {size_resnet}")

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((size_resnet, size_resnet)),
    transforms.ToTensor()
])

# Création des loaders
train_loader_complete, (train_loader, test_loader), dataset = make_loader(img_dir, data_file, transform)

batch_loader = InfiniteDataLoader(dataset) 


# Affichage des tailles des batchs
images, legendes, types = next(iter(train_loader_complete))
print(f"Taille batchs d'images {images.shape}")
print(f'Taille batchs légendes {legendes.shape}')
print(f'Taille batchs types {types.shape}')


# Définition du bloc de base ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride=1,dropout_rate=0.3):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*channels)
            )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Définition du modèle ResNet-50
class ResNet50(nn.Module):
    def __init__(self, block, num_blocks, num_classes=18,dropout_rate=0.5):
        super(ResNet50, self).__init__()
        self.dropout_rate = dropout_rate
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        #si on veut sortir avant la derinere couche, il faut enlever la ligne juste en dessous 
        #out = self.fc(out)
        return out

# Fonction pour initialiser ResNet-50
def resnet50():
    return ResNet50(BasicBlock, [3, 4, 6, 3])
# Ici on print le resultat pour voir si les tailles fit bien
model_pretrained = resnet50()
resultat = model_pretrained(images)
model_pretrained.to(device)
print(f'taille sortie {resultat.shape}') #torch.Size([32, 512])
print(f'taille attendue {types.shape}') #torch.Size([32])

# On charge si jamais on a deja entrainé, sinon on entraine
name = 'model_pretrained_6.pth'
isModel = True
if os.path.isfile(name):
    model_pretrained.load_state_dict(torch.load(name,map_location=torch.device(device) ))
    print("Successsfully loaded model pretrained")
else:
    print("Fail to load model pretrained")
    isModel = False

# Paramètres de l'entrainement du CNN
num_epochs = 100
lr = 1e-4
loss_function = nn.CrossEntropyLoss()
# Au cas ou pour eviter l'overfitting
weight_decay = 0.00
optimizer = optim.Adam(model_pretrained.parameters(), lr=lr, weight_decay=weight_decay)

# Boucle d'entrainement du CNN
if not isModel :
    for epoch in range(num_epochs):
        #training
        avg_train_loss = 0
        correct = 0
        total = 0
        for batch_idx , (images, _, types) in enumerate(train_loader_complete): 
            model_pretrained.train(True)
            optimizer.zero_grad()
            outputs_train = model_pretrained(images.to(device))
            loss_train = loss_function(outputs_train, types.to(device).long())
            loss_train.backward()
            optimizer.step()

            avg_train_loss += loss_train.item()
            model_pretrained.train(False)
            with torch.no_grad():
                _, predicted_train = torch.max(F.softmax(outputs_train).data, 1)
                total += types.size(0)
                correct += (predicted_train == types.to(device)).sum().item()

        avg_train_loss /= batch_idx+1
        print(f'Données pour le training')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f} , Accuracy : {(100 * correct / total):.2f} % ')
        
        #validation
        avg_val_loss = 0
        correct_val = 0
        total_val = 0
        model_pretrained.eval()
        with torch.no_grad():
            for images_val, _, types_val in test_loader:
                outputs_val = model_pretrained(images_val.to(device))
                loss_val = loss_function(outputs_val, types_val.to(device).long())
                avg_val_loss += loss_val.item()
                _, predicted_val = torch.max(F.softmax(outputs_val).data, 1)
                total_val += types_val.size(0)
                correct_val += (predicted_val == types_val.to(device)).sum().item()

        avg_val_loss /= len(test_loader)
        print(f'Données pour la validation')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {(100 * correct_val / total_val):.2f} % ')

        if epoch % 10 ==9 :
            # save le model
            torch.save(model_pretrained.state_dict(), 'model_pretrained_6.pth')




# Définition du modèle RNN
class RNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout_rate=0.4):
        super(RNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.bn = nn.BatchNorm1d(hidden_size)  
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)  # Ajout d'un dropout rate pour eviter l'overfitting
        self.to(device)

    def forward(self, captions):
        embedded = self.embedding(captions)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)  # Application des dropout
        out = self.fc(lstm_out)
        return out


# Définition du modèle combiné CNN-RNN
class CombinedModel(nn.Module):
    def __init__(self, cnn_model, rnn_model, vocab_size, dropout_rate = 0.3):
        super(CombinedModel, self).__init__()
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model
        self.fc = nn.Linear(512 + vocab_size, vocab_size)  # Ajustement des tailles
        self.dropout = nn.Dropout(dropout_rate)  

    def forward(self, images, captions):
        # Passage des images à travers le modèle CNN
        cnn_out = self.cnn_model(images)  # La sortie attendue est [batch_size, 512]

        # Passage des légendes à travers le modèle RNN
        rnn_out = self.rnn_model(captions)  # La sortie attendue est [batch_size, seq_length, vocab_size]

        # Ajout d'une dimension à cnn_out pour correspondre à rnn_out
        cnn_out = cnn_out.unsqueeze(1)  # [batch_size, 1, 512]

        # Répétition de cnn_out pour correspondre à la longueur de la séquence de rnn_out
        cnn_out = cnn_out.repeat(1, captions.size(1), 1)  # [batch_size, seq_length, 512]

        # Concaténation des sorties du CNN et du RNN
        combined_out = torch.cat((cnn_out, rnn_out), dim=2)  # [batch_size, seq_length, 512 + vocab_size]

        # Redimensionnement pour la couche fully connected
        combined_out = combined_out.view(-1, 512 + vocab_size)  # [batch_size * seq_length, 512 + vocab_size]
        combined_out = self.dropout(combined_out)  # Apply dropout before the fully connected layer

        # Passage à travers la couche fully connected finale
        final_out = self.fc(combined_out)  # [batch_size * seq_length, vocab_size]

        # Redimensionnement pour obtenir la sortie finale
        final_out = final_out.view(images.size(0), captions.size(1), -1)  # [batch_size, seq_length, vocab_size]

        return final_out

# Tailles des différents éléments
embedding_dim = 512  
hidden_size_rnn = 512  
num_classes_combined = 18
vocab_size = len(dataset.vocab)

#rnn_model
rnn_model = RNN_LSTM(vocab_size, embedding_dim, hidden_size_rnn, num_layers=1)

#combined_model
combined_model = CombinedModel(model_pretrained, rnn_model, vocab_size)
combined_model.to(device)

@torch.no_grad()
def generate(combined_model, ixs, image, dataset, temperature=1.1, top_k=10, repetition_penalty=2.5): #j'ai utilisé beacuoup de paramètres différents, plus temperature est elevé, plus il y a de randomness, et plsu top_k est elevé pareil.
    combined_model.eval()  # On set au mode eval
    for _ in range(dataset.max_length + 2):
        # On récupère les données de sorties
        output = combined_model(image, ixs)
        # On sélectionne la dernière étape
        logits = output[:,-1,:] 

        # Appliquer la pénalité de répétition
        for idx in range(logits.size(0)):
            for token in ixs[idx]:
                if token.item() in logits[idx]:
                    logits[idx][token.item()] /= repetition_penalty
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        # On utilise ici le top_k sampling, au lieu de max
        top_probs, top_ixs = probs.topk(top_k)
        next_word = top_ixs[:, torch.multinomial(top_probs, 1)]
        
        next_word = next_word.squeeze(1)
        # On ajoute le nouveau mot
        ixs = torch.cat((ixs, next_word), dim=-1)

    return ixs

@torch.no_grad()
def print_samples(combined_model, dataset, device):
    combined_model.eval()  # On set au mode eval

    # Initialisation avec le caractère <SOS>
    X_init = torch.tensor([[dataset.vocab.stoi["<SOS>"]]], dtype=torch.long).to(device)

    # On choisit aléatoirement une image
    random_index = random.randint(0, len(dataset) - 1)
    image, true_caption, _ = dataset[random_index]
    image = image.unsqueeze(0).to(device) 

    # Génère la caption
    generated_caption = generate(combined_model, X_init, image, dataset)

    # On décode la génération
    generated_words = dataset.vocab.decode(generated_caption.squeeze().tolist())
    true_words = dataset.vocab.decode(true_caption.tolist())

    # On enleve les tokens spéciaux
    generated_sentence = ' '.join([word for word in generated_words if word not in ('<SOS>', '<EOS>', '<PAD>', '<UNK>')])
    true_sentence = ' '.join([word for word in true_words if word not in ('<SOS>', '<EOS>', '<PAD>', '<UNK>')])

    print('Generated caption:', generated_sentence)
    print('True caption:', true_sentence)

# Entraînement du modèle combiné
# Paramètres de l'entrainement
lr = 1e-4
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(combined_model.parameters(), lr=lr, weight_decay=1e-5)
num_epochs =100
best_accuracy = 0.0  # Variable pour save le meilleur modèle


# Utilisation d'un scheduler pour améliorer les perf (j'essaye ici différentes choses que j'ai pu trouver sur internet pour avoir des meilleurs resultats)
#scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
name = 'best_model_7.pth'
isModel = True
if os.path.isfile(name):
    combined_model.load_state_dict(torch.load('best_model_7.pth'))
    print("Successfully loaded combined model")
else:
    print("Failed to load combined model")
    isModel = False
if not isModel : 
    for epoch in range(num_epochs):
        start_time = time.time()  # Début de la mesure du temps pour l'époque
        print(f'Start time: {start_time}')
        total_correct = 0
        total_samples = 0
        total_loss = 0
        combined_model.train()
        for images, captions, labels in train_loader:
            optimizer.zero_grad()
            # Passage des données à travers le modèle combiné
            outputs = combined_model(images.to(device), captions.to(device))
            outputs = outputs[:, 1:, :].contiguous().view(-1, outputs.size(-1))  # Ignorer le premier token et redimensionner
            targets = captions[:, 1:].contiguous().view(-1)  # Décalage et redimensionnement des cibles

            # Vérif que la taille du lot de `outputs` correspond à celle de `targets`.
            assert outputs.size(0) == targets.size(0), "Mismatch in batch size between outputs and targets"

            # Calcul de la perte avec la fonction de perte cross_entropy
            loss = loss_function(outputs, targets.to(device))
            # Rétropropagation et mise à jour des poids
            loss.backward()
            optimizer.step()

            # Calcul de la précision
            _, predicted_indices = torch.max(outputs, dim=1)
            total_correct += (predicted_indices == targets.to(device)).sum().item()
            total_samples += targets.size(0)
            end_time = time.time()  # Fin de la mesure du temps pour l'époque
            epoch_duration = end_time - start_time  # Calcul de la durée de l'époque
        
        #scheduler.step()
        train_accuracy = total_correct / total_samples
        print('Training')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.2%}')
        print(f"Duration of epoch: {epoch_duration:.2f} seconds")  # Affichage de la durée de l'époque
        print("Chaine de test pour le training")
        print_samples(combined_model , dataset, device)

        # Validation
        total_correct_val = 0
        total_samples_val = 0
        total_loss_val = 0

        combined_model.eval()
        with torch.no_grad():
            for images_val, captions_val, _ in test_loader:
                outputs_val = combined_model(images_val.to(device), captions_val.to(device))
                outputs_val = outputs_val[:, 1:, :].contiguous().view(-1, outputs_val.size(-1))  # Ignorer le premier token et redimensionner
                targets_val = captions_val[:, 1:].reshape(-1)  # On exclut <SOS> et on reshape 

                loss_val = loss_function(outputs_val, targets_val.to(device))
                total_loss_val += loss_val.item()

                _, predicted_indices_val = torch.max(outputs_val, dim=1)
                total_correct_val += (predicted_indices_val == targets_val.to(device)).sum().item()
                total_samples_val += targets_val.size(0)

                val_accuracy = total_correct_val / total_samples_val
                average_val_loss = total_loss_val / len(test_loader)
        val_accuracy = total_correct_val / total_samples_val
        average_val_loss = total_loss_val / len(test_loader)
        print(f'Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy (Train): {train_accuracy:.2%}, Accuracy (Val): {val_accuracy:.2%}')
        print("Chaine de test pour la validation")
        print_samples(combined_model , dataset, device)

        # On save que si le resultat est meilleur
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(combined_model.state_dict(), 'best_model_7.pth')

        combined_model.train() 

images, captions, types = next(iter(train_loader))
output = combined_model(images.to(device), captions.to(device))
output = output[:, 1:, :]  # Sélectionnez les 17 premiers éléments de la séquence

# On checke les tailles 
print(f'taille de sortie {output.shape}')
print(f'taille attendue {captions[:,1:].shape}')
print(f'output : {output.reshape(-1, output.size(-1)).shape} ')
print(f'légende : {captions[:,1:].reshape(-1).shape} ')

'''taille de sortie torch.Size([32, 17, 229])
taille attendue torch.Size([32, 17])
output : torch.Size([544, 229]) 
légende : torch.Size([544]) '''

'''
Exemple de ce que j'ai durant l'entrainement

Generated caption: an an an an an very very very very cute cute cute looking looking looking looking looking looking
True caption: a very cute looking bird with big wings

Generated caption: horse orange orange orange orange orange orange bat open flames flames flames flames flames flames flames flames flames
True caption: a picture of a white horse with orange and yellow flames
'''
