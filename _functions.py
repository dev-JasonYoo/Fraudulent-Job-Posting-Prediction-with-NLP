import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from matplotlib import pyplot as plt

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def custom_classification_report(y_true, y_pred, labels=[0,1], target_names=['Not Fraud', 'Fraud'], 
                                 sample_weight=None, digits=2, beta=2.0):
    """
    Minimal custom classification report with F-beta score
    Took sklearn.metrics.classification_report as the baseline and took out some details
    The original classification_report can be found here (https://github.com/scikit-learn/scikit-learn/blob/1eb422d6c5/sklearn/metrics/_classification.py#L2812)
    """
    
    # compute the metrics
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        beta=beta
    )

    # headers and widths
    headers = ["precision", "recall", f"f{beta:g}-score", "support"]
    name_width = max(len(n) for n in target_names)
    width = max(name_width, len(headers[2]), digits)

    # structure the layout
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
    
    for i, name in enumerate(target_names):
        report += row_fmt.format(name, p[i], r[i], f[i], s[i], width=width, digits=digits)

    report += "\n"

    # Construct the accuracy row
    accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    total_support = np.sum(s)
    
    row_fmt_acc = "{:>{width}s} " + " {:>9}" * 2 + " {:>9.{digits}f}" + " {:>9}\n"
    report += row_fmt_acc.format("accuracy", "", "", accuracy, total_support, width=width, digits=digits)

    return report