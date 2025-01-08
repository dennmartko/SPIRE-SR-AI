import matplotlib.pyplot as plt

def data_debug_plot(inputs, mask_plus_targets, input_class_names, target_class_names, save, num_samples=5):
    """
    Plots a grid of images for the input classes and target classes.

    Args:
        inputs (numpy.ndarray): Input array of shape (num_samples, height, width, num_classes).
        labels (numpy.ndarray): Label array of shape (num_samples, height, width).
        input_class_names (list): List of input class names.
        target_class_name (list): List containing the target class name(s).
        num_samples (int): Number of samples to plot (default is 5).
    """
    plt.figure(figsize=(10, 10))
    count = 0
    mask_plus_targets_classes = target_class_names + [cl + "_mask" for cl in target_class_names]
    classes = input_class_names + mask_plus_targets_classes
    for i in range(num_samples):
        s = 0
        for cls_i, cls in enumerate(classes):
            plt.subplot(num_samples, len(classes), count + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            if cls in mask_plus_targets_classes:
                plt.imshow(mask_plus_targets[i, :, :, s], vmin=0)
                s+=1
            else:
                plt.imshow(inputs[i, :, :, cls_i], vmin=0)
            plt.xlabel(cls)
            count += 1
    plt.tight_layout(w_pad=0.05)
    plt.savefig(save, dpi=300)
    plt.close()

def display_predictions(inputs, targets, predictions, input_class_names, target_class_names, save, num_samples=5):
    """
    Plots a grid of the input, target, and predicted images.
    """
    plt.figure(figsize=(10, 10))
    count = 0
    prediction_labels = [f"predicted {cl}" for cl in target_class_names]
    classes = input_class_names + target_class_names + prediction_labels
    
    for i in range(num_samples):
        c_pred = 0
        c_target = 0
        c_input = 0  # Separate counter for input channels
        
        for cls in classes:
            plt.subplot(num_samples, len(classes), count + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            
            if cls in prediction_labels:
                plt.imshow(predictions[i, :, :, c_pred], vmin=0, vmax=10e-3)
                c_pred += 1
            elif cls in target_class_names:
                plt.imshow(targets[i, :, :, c_target], vmin=0, vmax=10e-3)
                c_target += 1
            else:  # cls is in input_class_names
                plt.imshow(inputs[i, :, :, c_input], vmin=0)
                c_input += 1  # Increment input channel counter
            
            plt.xlabel(cls)
            count += 1

    plt.tight_layout(w_pad=0.05)
    plt.savefig(save, dpi=300)
    plt.close()

def plot_history(history, save):
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.title("Train and Validation loss", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(save, dpi=300)
    plt.close()