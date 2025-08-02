import torch
from torchvision import transforms
from PIL import Image
import json
from train_vqa_model import VizWizDataset, VQAModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("answer_to_index.json", "r") as f:
    answer_to_index = json.load(f)

with open("index_to_answer.json", "r") as f:
    index_to_answer = json.load(f)


dummy_dataset = VizWizDataset(
    data_dir="VizWiz_VQA/train",
    annotation_file="VizWiz_VQA/Annotations/train.json",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)


model = VQAModel(
    vocab_size=len(dummy_dataset.word2idx),
    num_classes=len(answer_to_index)
).to(device)

model.load_state_dict(torch.load("vqa_model.pth", map_location=device))
model.eval()


image_path = input("Enter image path (e.g. VizWiz_VQA/val/VizWiz_val_00000017.jpg): ").strip().strip('"')
question_text = input("Enter your question: ").strip()


image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0).to(device)


tokens = question_text.lower().split()
question_indices = [dummy_dataset.word2idx.get(token, dummy_dataset.word2idx["<unk>"]) for token in tokens]
question_tensor = torch.tensor(question_indices).unsqueeze(0).to(device)


with torch.no_grad():
    output = model(image_tensor, question_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)

    # Top-1 prediction
    predicted_index = probs.argmax(dim=1).item()
    predicted_answer = index_to_answer.get(str(predicted_index), "unsuitable")
    print(f"\nPredicted Answer: {predicted_answer}")

    # Top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, 5)
    print("\nTop-5 Predictions:")
    for i in range(5):
        idx = top5_indices[0][i].item()
        prob = top5_probs[0][i].item()
        ans = index_to_answer.get(str(idx), "unknown")
        print(f"{i+1}. {ans} ({prob:.4f})")