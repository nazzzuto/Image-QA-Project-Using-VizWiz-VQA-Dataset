import os
import json
import nltk
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm


nltk.download('all')

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  

def preprocess_question(question, word2idx, max_question_length=20):
    tokens = word_tokenize(question.lower())
    indices = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    if len(indices) < max_question_length:
        indices += [word2idx['<pad>']] * (max_question_length - len(indices))
    else:
        indices = indices[:max_question_length]
    return torch.tensor(indices).unsqueeze(0)  



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VizWizDataset(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None, max_q_len=20):
        with open(annotation_file, 'r', encoding='utf-8', errors='replace') as f:
            self.annotations = json.load(f)

        self.data_dir = data_dir
        self.transform = transform
        self.max_q_len = max_q_len
        self.tokenizer = word_tokenize

        print(" Building vocabulary...")
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.build_vocab()

        print(" Building answer vocabulary...")
        self.answer2idx = {}
        self.idx2answer = {}
        self.build_answer_vocab()

    def build_vocab(self):
        counter = Counter()
        for item in self.annotations:
            question = item['question'].lower()
            tokens = self.tokenizer(question)
            counter.update(tokens)

        for word in counter:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def build_answer_vocab(self):
        counter = Counter()
        for item in self.annotations:
            for ans in item['answers']:
                counter[ans['answer'].lower()] += 1

        most_common = counter.most_common(1000)
        for idx, (ans, _) in enumerate(most_common):
            self.answer2idx[ans] = idx
            self.idx2answer[idx] = ans

    def encode_question(self, question):
        tokens = self.tokenizer(question.lower())
        indices = [self.word2idx.get(token, 1) for token in tokens]
        if len(indices) < self.max_q_len:
            indices += [0] * (self.max_q_len - len(indices))
        else:
            indices = indices[:self.max_q_len]
        return torch.tensor(indices)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        image_filename = item['image']
        image_path = os.path.join(self.data_dir, image_filename)

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            print(f" Could not open image: {image_path} | Skipping...")
            image = Image.new('RGB', (224, 224))  # dummy image

        if self.transform:
            image = self.transform(image)

        question = self.encode_question(item['question'])
        answer = item['answers'][0]['answer'].lower()
        label = self.answer2idx.get(answer, 1)  
        return image, question, torch.tensor(label)

# MODEL DEFINITION

class VQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_classes=1000):
        super(VQAModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, question):
        with torch.no_grad():
            img_feat = self.cnn(image)

        embed = self.embedding(question)
        _, hidden = self.rnn(embed)
        hidden = hidden.squeeze(0)

        combined = torch.cat((img_feat, hidden), dim=1)
        out = self.classifier(combined)
        return out


# VALIDATION FUNCTION

def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, questions, labels in tqdm(val_loader, desc="Validating"):
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            outputs = model(images, questions)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f" Validation Accuracy: {accuracy:.2f}%")

# PREDICTION FUNCTION

def predict_answer(model, dataset, image_path, question_text):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    tokens = dataset.encode_question(question_text).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image, tokens)
        _, predicted = outputs.max(1)

    answer = dataset.idx2answer[predicted.item()]
    print(f" Predicted Answer: {answer}")
    return answer


# TRAINING FUNCTION

def train():
    print(" Preparing transforms and datasets...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = VizWizDataset(
        data_dir="VizWiz_VQA/train",
        annotation_file="VizWiz_VQA/Annotations/train.json",
        transform=transform
    )

    val_dataset = VizWizDataset(
        data_dir="VizWiz_VQA/val",
        annotation_file="VizWiz_VQA/Annotations/val.json",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(" Initializing model...")
    model = VQAModel(vocab_size=len(train_dataset.word2idx), num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n Starting training...\n")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for images, questions, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)

            outputs = model(images, questions)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f" Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
        validate(model, val_loader)

    torch.save(model.state_dict(), "vqa_model.pth")
    print("\n Model saved as 'vqa_model.pth'")

   

    # Save answer 
    import json
    with open("answer_to_index.json", "w") as f:
        json.dump(train_dataset.answer2idx, f)

    with open("index_to_answer.json", "w") as f:
        json.dump(train_dataset.idx2answer, f)
        
    print("mapping saved!")

# RUN TRAINING

if __name__ == "_main_":
    print(" Starting training...")
    try:
        train()
    except Exception as e:
        print(" Error during training:", str(e))