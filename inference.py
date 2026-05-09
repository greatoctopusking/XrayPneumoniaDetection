import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_test_loader
from utils.model import build_efficientnetb3

DATA_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\data'
MODELS_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\models'
RESULTS_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_preds = []
    all_ids = []

    for images, ids in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_ids.extend(ids)

    return all_preds, all_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficientnetb3_scenario1.pth',
                        help='Model filename in models/ directory')
    parser.add_argument('--output', type=str, default='submission_efficientnet.csv',
                        help='Output CSV filename')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    args = parser.parse_args()

    model_path = os.path.join(MODELS_DIR, args.model)
    if not os.path.exists(model_path):
        alt_path = os.path.join(MODELS_DIR, 'efficientnetb3_scenario1.pth')
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f'Model not found: {model_path}')
            print(f'Available models: {os.listdir(MODELS_DIR)}')
            sys.exit(1)

    print(f'Loading model: {model_path}')
    model = build_efficientnetb3(num_classes=2, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print(f'Device: {DEVICE}')

    print('Loading test data...')
    test_loader, test_ids = get_test_loader(batch_size=args.batch_size)
    print(f'Test samples: {len(test_ids)}')

    print('Making predictions...')
    predictions, ids = predict(model, test_loader)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, args.output)
    with open(output_path, 'w') as f:
        f.write('id,TARGET\n')
        for img_id, pred in zip(ids, predictions):
            f.write(f'{img_id},{pred}\n')

    normal_count = sum(1 for p in predictions if p == 0)
    pneumonia_count = sum(1 for p in predictions if p == 1)
    print(f'\nSubmission saved: {output_path}')
    print(f'Predictions: Normal={normal_count}, Pneumonia={pneumonia_count}')


if __name__ == '__main__':
    main()
