"""
Evaluate PathMNIST models on Kather 2016 dataset (population shift evaluation).

This script:
1. Loads a trained PathMNIST model (ResNet18 standard)
2. Evaluates on Kather 2016 CRC texture dataset
3. Computes accuracy, balanced accuracy
4. Generates confusion matrix
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report
import sys

# Add paths for imports
script_path = Path(__file__).parent
sys.path.insert(0, str(script_path))

from Data.kather_texture_dataset_6class import (
    KatherTexture2016,
    create_kather_dataloader,
    collapse_pathmnist_to_6class,
    COLLAPSED_6CLASS,
    print_taxonomy_info
)
from torchvision import models as torch_models
import torch.nn as nn


def evaluate_model_on_kather(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    verbose: bool = True
):
    """
    Evaluate PathMNIST model on Kather dataset.
    
    Args:
        model: Trained PathMNIST model (9-class output)
        dataloader: Kather dataset dataloader
        device: Device to run on
        verbose: Print progress
    
    Returns:
        dict: Dictionary with predictions, labels, and metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if verbose and batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx+1}/{len(dataloader)}")
            
            images = images.to(device)
            
            # Forward pass (9-class logits)
            logits = model(images)
            
            # Collapse to 6-class probabilities
            probs_6class = collapse_pathmnist_to_6class(logits, is_logits=True)
            
            # Get predictions
            predictions = probs_6class.argmax(dim=1)
            
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs_6class.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    accuracy: float,
    balanced_accuracy: float,
    model_name: str = 'ResNet18',
    dataset_name: str = 'Kather 2016',
    save_path: Path = None
):
    """
    Plot confusion matrix with metrics.
    
    Args:
        cm: Confusion matrix (N_classes x N_classes)
        class_names: List of class names
        accuracy: Overall accuracy
        balanced_accuracy: Balanced accuracy
        model_name: Model name for title
        dataset_name: Dataset name for title
        save_path: Path to save figure (if None, just display)
    """
    # Normalize confusion matrix by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot raw counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=class_names, yticklabels=class_names,
        ax=ax1, cbar_kws={'label': 'Count'}
    )
    ax1.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax1.set_title(f'Confusion Matrix - Counts\n{model_name} on {dataset_name}', 
                  fontsize=14, fontweight='bold')
    
    # Plot normalized (percentages)
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
        xticklabels=class_names, yticklabels=class_names,
        ax=ax2, cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1
    )
    ax2.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax2.set_title(f'Confusion Matrix - Normalized\n{model_name} on {dataset_name}', 
                  fontsize=14, fontweight='bold')
    
    # Add metrics text
    metrics_text = f"Accuracy: {accuracy:.2%}\nBalanced Accuracy: {balanced_accuracy:.2%}"
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=14, 
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return fig


def main():
    """Main evaluation script."""
    
    print("="*80)
    print("PathMNIST → Kather 2016 Population Shift Evaluation")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_root = script_dir / 'Data' / 'kather_texture' / 'Kather_texture_2016_image_tiles_5000'
    models_dir = script_dir / 'models' / '224*224'
    output_dir = script_dir / 'kather_evaluation_results'
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    flag = 'pathmnist'
    model_name = 'resnet18'
    setup = 'randaug0'  # Use 'randaug0' for standard, 'randaug0_dropout03' for DO
    fold = 0  # Which CV fold to use
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {flag}")
    print(f"  Model: {model_name}")
    print(f"  Setup: {setup}")
    print(f"  Fold: {fold}")
    print(f"  Data root: {data_root}")
    print(f"  Models dir: {models_dir}")
    
    # Print taxonomy info
    print("\n" + "="*80)
    print_taxonomy_info()
    
    # Load dataset
    print("\n" + "="*80)
    print("Loading Kather 2016 Dataset")
    print("="*80)
    
    dataloader = create_kather_dataloader(
        data_root=str(data_root),
        batch_size=64,
        num_workers=4,
        target_size=224
    )
    
    dataset = dataloader.dataset
    print(f"\nDataset loaded: {len(dataset)} images")
    print(f"Class distribution:")
    for class_name in COLLAPSED_6CLASS:
        count = dataset.get_class_distribution().get(class_name, 0)
        print(f"  {class_name:8s}: {count:5d} images ({count/len(dataset)*100:.1f}%)")
    
    # Load model(s)
    print("\n" + "="*80)
    print("Loading PathMNIST Model(s)")
    print("="*80)
    
    try:
        # Build model architecture (PathMNIST has 9 classes)
        num_classes = 9
        
        if fold == 'ensemble':
            # Load all 5 folds for ensemble
            models = []
            for fold_idx in range(5):
                if model_name == 'resnet18':
                    model = torch_models.resnet18(weights=None)
                    model.fc = nn.Linear(model.fc.in_features, num_classes)
                elif model_name == 'vit_b_16':
                    model = torch_models.vit_b_16(weights=None)
                    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                
                model_path = models_dir / f'{flag}_{model_name}_224_{setup}_fold_{fold_idx}.pt'
                
                if not model_path.exists():
                    print(f"\n❌ Model file not found: {model_path}")
                    print(f"   Cannot create ensemble - missing fold {fold_idx}")
                    return
                
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model = model.to(device)
                model.eval()
                models.append(model)
                print(f"  ✓ Loaded fold {fold_idx}: {model_path.name}")
            
            print(f"\n✓ Ensemble of {len(models)} models loaded successfully")
            model = models  # Pass list of models to evaluation
        else:
            # Load single fold
            if model_name == 'resnet18':
                model = torch_models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'vit_b_16':
                model = torch_models.vit_b_16(weights=None)
                model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            model_path = models_dir / f'{flag}_{model_name}_224_{setup}_fold_{fold}.pt'
            
            print(f"\nLoading model from: {model_path}")
            
            if not model_path.exists():
                print(f"\n❌ Model file not found: {model_path}")
                print(f"\nℹ️  PathMNIST models need to be trained first.")
                print(f"   Available datasets in {models_dir}:")
                available = list(models_dir.glob('*_resnet18_224_*.pt'))
                if available:
                    datasets = set([f.name.split('_')[0] for f in available[:10]])
                    for d in sorted(datasets):
                        print(f"     - {d}")
                    print(f"\n   To evaluate Kather with one of these, change 'flag' in the script.")
                else:
                    print(f"     No trained models found!")
                return
            
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            
            print(f"✓ Model loaded successfully ({model_name})")
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print(f"Expected model path: {models_dir / flag / 'fold_0.pth'}")
        return
    
    # Evaluate
    print("\n" + "="*80)
    print("Running Evaluation")
    print("="*80)
    
    results = evaluate_model_on_kather(model, dataloader, device, verbose=True)
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:          {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
    
    # Per-class accuracy
    cm = results['confusion_matrix']
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\nPer-Class Accuracy:")
    for idx, class_name in enumerate(COLLAPSED_6CLASS):
        print(f"  {class_name:8s}: {per_class_acc[idx]:.4f} ({per_class_acc[idx]*100:.2f}%)")
    
    # Print classification report
    print("\n" + "="*80)
    print("Classification Report")
    print("="*80)
    print(classification_report(
        results['labels'], 
        results['predictions'],
        target_names=COLLAPSED_6CLASS,
        digits=4
    ))
    
    # Plot confusion matrix
    print("\n" + "="*80)
    print("Generating Confusion Matrix")
    print("="*80)
    
    save_path = output_dir / f'confusion_matrix_{flag}_{model_name}_{setup}_kather2016.png'
    
    plot_confusion_matrix(
        cm=results['confusion_matrix'],
        class_names=COLLAPSED_6CLASS,
        accuracy=results['accuracy'],
        balanced_accuracy=results['balanced_accuracy'],
        model_name=f'{model_name.upper()} ({setup})',
        dataset_name='Kather 2016 CRC Texture',
        save_path=save_path
    )
    
    # Save results
    fold_str = 'ensemble' if fold == 'ensemble' else f'fold{fold}'
    results_path = output_dir / f'evaluation_results_{flag}_{model_name}_{setup}_{fold_str}_kather2016_6class.npz'
    np.savez(
        results_path,
        predictions=results['predictions'],
        labels=results['labels'],
        probabilities=results['probabilities'],
        confusion_matrix=results['confusion_matrix'],
        accuracy=results['accuracy'],
        balanced_accuracy=results['balanced_accuracy'],
        class_names=COLLAPSED_6CLASS
    )
    print(f"✓ Results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("✓ Evaluation Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
