import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_train_val_loaders, get_final_test_loader
from utils.model import build_efficientnetb3, unfreeze_backbone
from utils.evaluate import compute_metrics, print_metrics, plot_confusion_matrix, plot_training_curves

DATA_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\data'
MODELS_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\models'
RESULTS_DIR = r'D:\GithubRepositories\XrayPneumoniaDetection\results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCENARIOS = {
    1: {'epochs': 20, 'lr': 0.001, 'batch_size': 16, 'optimizer': 'adamax'},
    2: {'epochs': 30, 'lr': 0.001, 'batch_size': 20, 'optimizer': 'adamax'},
    3: {'epochs': 30, 'lr': 0.01, 'batch_size': 45, 'optimizer': 'adamax'},
}


def get_optimizer(opt_name, params, lr):
    if opt_name == 'adamax':
        return optim.Adamax(params, lr=lr)
    elif opt_name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9)
    elif opt_name == 'rmsprop':
        return optim.RMSprop(params, lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {opt_name}')


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train_scenario(model, train_loader, val_loader, scenario_config, scenario_id):
    epochs = scenario_config['epochs']
    lr = scenario_config['lr']
    opt_name = scenario_config['optimizer']

    print(f'\n{"="*60}')
    print(f'Scenario {scenario_id}: {opt_name}, lr={lr}, epochs={epochs}, bs={scenario_config["batch_size"]}')
    print(f'Device: {DEVICE}')
    print(f'{"="*60}')

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(opt_name, filter(lambda p: p.requires_grad, model.parameters()), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_f1 = 0
    best_model_state = None
    patience = 8
    no_improve = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        val_metrics = compute_metrics(val_labels, val_preds)
        scheduler.step(val_metrics['f1_score'])

        print(f'Epoch {epoch+1:2d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'Val F1: {val_metrics["f1_score"]:.4f}')

        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_val_loss, final_val_acc, final_preds, final_labels = validate(model, val_loader, criterion)
    final_metrics = compute_metrics(final_labels, final_preds)

    save_name = f'efficientnetb3_scenario{scenario_id}.pth'
    save_path = os.path.join(MODELS_DIR, save_name)
    torch.save(best_model_state, save_path)
    print(f'\nSaved best model: {save_path}')

    return final_metrics, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3], default=1,
                        help='Training scenario (1/2/3)')
    parser.add_argument('--unfreeze_after', type=int, default=0,
                        help='Unfreeze last N blocks after training head')
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config = SCENARIOS[args.scenario]
    print(f'Loading data with batch_size={config["batch_size"]}...')
    train_loader, val_loader = get_train_val_loaders(batch_size=config['batch_size'])
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

    model = build_efficientnetb3(num_classes=2, freeze_backbone=True).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal params: {total_params:,} | Trainable: {trainable_params:,}')

    print(f'\n--- Phase 1: Training classification head ---')
    metrics, history = train_scenario(model, train_loader, val_loader, config, args.scenario)
    print_metrics(metrics, prefix='Phase 1 (Head only)')

    curves_path = os.path.join(RESULTS_DIR, f'training_curves_s{args.scenario}.png')
    plot_training_curves(history, curves_path)
    print(f'Training curves saved: {curves_path}')

    if args.unfreeze_after > 0:
        print(f'\n--- Phase 2: Fine-tuning last {args.unfreeze_after} blocks ---')
        unfreeze_backbone(model, depth=args.unfreeze_after)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable after unfreeze: {trainable_params:,}')

        ft_config = {**config, 'lr': config['lr'] * 0.1}
        ft_metrics, ft_history = train_scenario(model, train_loader, val_loader, ft_config, f'{args.scenario}_ft')
        print_metrics(ft_metrics, prefix='Phase 2 (Fine-tuned)')

        curves_path = os.path.join(RESULTS_DIR, f'training_curves_s{args.scenario}_ft.png')
        plot_training_curves(ft_history, curves_path)

    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_val_s{args.scenario}.png')
    plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
    print(f'Val confusion matrix saved: {cm_path}')

    print(f'\n{"="*60}')
    print('Final evaluation on holdout test set (data/val/)')
    print(f'{"="*60}')
    test_loader, test_count = get_final_test_loader(batch_size=config['batch_size'])
    if test_loader is not None:
        test_criterion = nn.CrossEntropyLoss()
        _, _, test_preds, test_labels = validate(model, test_loader, test_criterion)
        test_metrics = compute_metrics(test_labels, test_preds)
        print_metrics(test_metrics, prefix='Holdout Test')

        cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_test_s{args.scenario}.png')
        plot_confusion_matrix(test_metrics['confusion_matrix'], cm_path)
        print(f'Test confusion matrix saved: {cm_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
