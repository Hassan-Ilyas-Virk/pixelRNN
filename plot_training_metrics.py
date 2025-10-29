import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set light mode style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

def plot_training_metrics(log_file_path='checkpoints/training_log.txt'):
    """
    Plot training metrics (L1 Loss, SSIM, Total Loss) against epochs.
    
    Args:
        log_file_path (str): Path to the training log file
    """
    
    # Read the training log
    try:
        # Read the tab-separated values file
        df = pd.read_csv(log_file_path, sep='\t')
        
        # Extract data
        epochs = df['Epoch'].values
        l1_loss = df['L1_Loss'].values
        ssim = df['SSIM'].values
        total_loss = df['Total_Loss'].values
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Over Epochs', fontsize=16, fontweight='bold')
        
        # Plot 1: L1 Loss vs Epochs
        axes[0, 0].plot(epochs, l1_loss, color='#2E86AB', linewidth=2.5, marker='o', markersize=4, markerfacecolor='#2E86AB', markeredgecolor='white', markeredgewidth=1)
        axes[0, 0].set_title('L1 Loss vs Epochs', fontsize=12, fontweight='bold', color='#2C3E50')
        axes[0, 0].set_xlabel('Epoch', fontsize=11, color='#2C3E50')
        axes[0, 0].set_ylabel('L1 Loss', fontsize=11, color='#2C3E50')
        axes[0, 0].grid(True, alpha=0.3, color='gray')
        axes[0, 0].set_xlim(0, max(epochs))
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        
        # Plot 2: SSIM vs Epochs
        axes[0, 1].plot(epochs, ssim, color='#A23B72', linewidth=2.5, marker='s', markersize=4, markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=1)
        axes[0, 1].set_title('SSIM vs Epochs', fontsize=12, fontweight='bold', color='#2C3E50')
        axes[0, 1].set_xlabel('Epoch', fontsize=11, color='#2C3E50')
        axes[0, 1].set_ylabel('SSIM', fontsize=11, color='#2C3E50')
        axes[0, 1].grid(True, alpha=0.3, color='gray')
        axes[0, 1].set_xlim(0, max(epochs))
        axes[0, 1].set_ylim(0.8, 1.0)  # SSIM typically ranges from 0 to 1
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        # Plot 3: Total Loss vs Epochs
        axes[1, 0].plot(epochs, total_loss, color='#F18F01', linewidth=2.5, marker='^', markersize=4, markerfacecolor='#F18F01', markeredgecolor='white', markeredgewidth=1)
        axes[1, 0].set_title('Total Loss vs Epochs', fontsize=12, fontweight='bold', color='#2C3E50')
        axes[1, 0].set_xlabel('Epoch', fontsize=11, color='#2C3E50')
        axes[1, 0].set_ylabel('Total Loss', fontsize=11, color='#2C3E50')
        axes[1, 0].grid(True, alpha=0.3, color='gray')
        axes[1, 0].set_xlim(0, max(epochs))
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
        
        # Plot 4: All metrics combined (normalized for comparison)
        # Normalize metrics to 0-1 range for comparison
        l1_norm = (l1_loss - np.min(l1_loss)) / (np.max(l1_loss) - np.min(l1_loss))
        ssim_norm = (ssim - np.min(ssim)) / (np.max(ssim) - np.min(ssim))
        total_norm = (total_loss - np.min(total_loss)) / (np.max(total_loss) - np.min(total_loss))
        
        axes[1, 1].plot(epochs, l1_norm, 'b-', linewidth=2, label='L1 Loss (normalized)', alpha=0.8)
        axes[1, 1].plot(epochs, ssim_norm, 'g-', linewidth=2, label='SSIM (normalized)', alpha=0.8)
        axes[1, 1].plot(epochs, total_norm, 'r-', linewidth=2, label='Total Loss (normalized)', alpha=0.8)
        axes[1, 1].set_title('All Metrics (Normalized) vs Epochs', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Normalized Value')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, max(epochs))
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('training_metrics_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved as 'training_metrics_plot.png'")
        
        # Show the plot
        plt.show()
        
        # Print summary statistics
        print("\n=== Training Summary ===")
        print(f"Total Epochs: {len(epochs)}")
        print(f"Final L1 Loss: {l1_loss[-1]:.6f}")
        print(f"Final SSIM: {ssim[-1]:.6f}")
        print(f"Final Total Loss: {total_loss[-1]:.6f}")
        print(f"Best SSIM: {np.max(ssim):.6f} (Epoch {epochs[np.argmax(ssim)]})")
        print(f"Lowest L1 Loss: {np.min(l1_loss):.6f} (Epoch {epochs[np.argmin(l1_loss)]})")
        print(f"Lowest Total Loss: {np.min(total_loss):.6f} (Epoch {epochs[np.argmin(total_loss)]})")
        
    except FileNotFoundError:
        print(f"Error: Could not find the training log file at '{log_file_path}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error reading or processing the training log: {e}")

def plot_individual_metrics(log_file_path='checkpoints/training_log.txt'):
    """
    Create separate individual plots for each metric.
    
    Args:
        log_file_path (str): Path to the training log file
    """
    
    try:
        # Read the training log
        df = pd.read_csv(log_file_path, sep='\t')
        
        epochs = df['Epoch'].values
        l1_loss = df['L1_Loss'].values
        ssim = df['SSIM'].values
        total_loss = df['Total_Loss'].values
        
        # Create individual plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # L1 Loss plot
        axes[0].plot(epochs, l1_loss, 'b-', linewidth=2, marker='o', markersize=2)
        axes[0].set_title('L1 Loss vs Epochs', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('L1 Loss', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, max(epochs))
        
        # SSIM plot
        axes[1].plot(epochs, ssim, 'g-', linewidth=2, marker='s', markersize=2)
        axes[1].set_title('SSIM vs Epochs', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('SSIM', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, max(epochs))
        axes[1].set_ylim(0.8, 1.0)
        
        # Total Loss plot
        axes[2].plot(epochs, total_loss, 'r-', linewidth=2, marker='^', markersize=2)
        axes[2].set_title('Total Loss vs Epochs', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Total Loss', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0, max(epochs))
        
        plt.tight_layout()
        plt.savefig('individual_metrics_plot.png', dpi=300, bbox_inches='tight')
        print("Individual metrics plot saved as 'individual_metrics_plot.png'")
        plt.show()
        
    except Exception as e:
        print(f"Error creating individual plots: {e}")

if __name__ == "__main__":
    print("Creating training metrics plots...")
    
    # Create the main combined plot
    plot_training_metrics()
    
    print("\n" + "="*50)
    
    # Create individual plots
    plot_individual_metrics()
    
    print("\nPlotting completed!")
