import argparse, os, numpy as np, torch
from lime import lime_image
from PIL import Image
from torchvision import datasets, transforms
from model import SimpleCNN
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='outputs/lime_heatmaps')
parser.add_argument('--image_index', type=int, default=0)
parser.add_argument('--num_classes', type=int, required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
dataset = datasets.ImageFolder(args.data_dir, transform=transform)
image_tensor, _ = dataset[args.image_index]
image_np = np.transpose(image_tensor.numpy(), (1, 2, 0))

model = SimpleCNN(num_classes=args.num_classes).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

def batch_predict(images):
    batch = torch.stack([transform(Image.fromarray((img * 255).astype(np.uint8)))
                         for img in images]).to(device)
    with torch.no_grad():
        return torch.nn.functional.softmax(model(batch), dim=1).cpu().numpy()

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image_np, batch_predict, top_labels=1,
                                         hide_color=0, num_samples=1000)
os.makedirs(args.output_dir, exist_ok=True)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                            positive_only=True, num_features=5,
                                            hide_rest=False)
plt.imshow(mark_boundaries(temp, mask))
plt.axis('off')
plt.savefig(os.path.join(args.output_dir, f"lime_{args.image_index}.png"))
