import json
from train_vqa_model import VizWizDataset
from torchvision import transforms

train_dataset = VizWizDataset(
    data_dir="VizWiz_VQA/train",
    annotation_file="VizWiz_VQA/Annotations/train.json",
    transform=transforms.Compose([])
)

# Save mappings
with open("answer_to_index.json", "w") as f:
    json.dump(train_dataset.answer2idx, f)

with open("index_to_answer.json", "w") as f:
    json.dump(train_dataset.idx2answer, f)

print(" Mappings saved!")