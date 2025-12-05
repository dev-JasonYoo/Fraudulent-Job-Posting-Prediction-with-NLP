import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from matplotlib import pyplot as plt

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    fig, ax = plt.subplots()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-fraudulent', 'Fraudulent'],
                yticklabels=['Non-fraudulent', 'Fraudulent'])
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig, ax

def custom_classification_report(y_true, y_pred, labels=[0,1], target_names=['Not Fraud', 'Fraud'], 
                                 sample_weight=None, digits=2, beta=2.0):
    """
    Minimal custom classification report with F-beta score
    Took sklearn.metrics.classification_report as the baseline and took out some details
    The original classification_report can be found here (https://github.com/scikit-learn/scikit-learn/blob/1eb422d6c5/sklearn/metrics/_classification.py#L2812)
    """
    
    # compute the metrics
    p, r, fbeta, s = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        beta=beta
    )

    # compute the f1 score
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        beta=1
    )

    # headers and widths
    headers = ["precision", "recall", f"f1-score", f"f{beta:g}-score", "support"]
    name_width = max(len(n) for n in target_names)
    width = max(name_width, len(headers[2]), digits)

    # structure the layout
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 4 + " {:>9}\n"
    
    for i, name in enumerate(target_names):
        report += row_fmt.format(name, p[i], r[i], f1[i], fbeta[i], s[i], width=width, digits=digits)

    report += "\n"

    # Construct the accuracy row
    accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    total_support = np.sum(s)
    
    row_fmt_acc = "{:>{width}s} " + " {:>9}" * 3 + " {:>9.{digits}f}" + " {:>9}\n"
    report += row_fmt_acc.format("accuracy", "", "", "", accuracy, total_support, width=width, digits=digits)

    return report


# for classifiers with thresholds

def get_max_fscore_i(precision, recall, beta=2.0):
    fbeta_scores = ((1+beta**2) * precision * recall) / ((beta**2 * precision) + recall)
    return np.argmax(fbeta_scores)

def plot_PR_curve(precision, recall, title='Precision-Recall Curve'):
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()

    plt.title(title)
    plt.plot(recall, precision, marker='.', label='PR curve')
    plt.tight_layout()
    plt.legend()
    plt.show()

    return fig, ax

def plot_PR_threshold_curve(precision, recall, thresholds, beta=2.0, title='Precision-Recall vs. Threshold Curve'):
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()

    max_score_i = get_max_fscore_i(precision, recall, beta=beta)

    plt.title(title)
    plt.plot(thresholds, precision[:-1], marker='.', label='Precision')
    plt.plot(thresholds, recall[:-1], marker='.', label='Recall')
    plt.axvline(x=thresholds[max_score_i], color='red', linestyle='--', label='Threshold at the maximum F-{beta:g} score')
    plt.tight_layout()
    plt.legend()
    plt.show()

    return fig, ax

# exclusively for neural networks

def plot_loss(loss_history, title, dont_show=False):
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    n_epoch = len(loss_history['loss'])
    if n_epoch < 50:
        plt.xticks((range(1, len(loss_history['loss'])+1, 5)))
    elif n_epoch < 100:
        plt.xticks((range(1, len(loss_history['loss'])+1, 10)))
    elif n_epoch < 300:
        plt.xticks((range(1, len(loss_history['loss'])+1, 30)))
    
    plt.plot(loss_history['loss'], label='Training Loss')
    plt.plot(loss_history['val_loss'], label='Validation Loss')
    plt.tight_layout()
    plt.legend()
    if not dont_show:
        plt.show()

    return fig, ax

def plot_learning_rate(lr_history, title):
    plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots()

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.xticks((range(1, len(lr_history)+1, 5)))
    plt.yticks([0.001 * (1/3)**i for i in range(4)])
    ax.set_yticklabels([f"{0.001 * (1/3)**i:.2e}" for i in range(4)])
    
    plt.plot(lr_history, label='Learning Rate')
    plt.tight_layout()
    plt.legend()
    plt.show()

    return fig, ax

def get_best_threshold_i(precision, recall, beta=2.0):
    fbeta_scores = ((1+beta**2) * precision * recall) / ((beta**2 * precision) + recall)
    return np.argmax(fbeta_scores)