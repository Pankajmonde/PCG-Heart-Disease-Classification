import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

class ModelEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def print_classification_report(self):
        report = classification_report(self.y_true, self.y_pred)
        print(report)

    def plot_roc_curve(self, y_scores):
        fpr, tpr, _ = roc_curve(self.y_true, y_scores)
        plt.plot(fpr, tpr, color='blue')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def calculate_auc(self, y_scores):
        return roc_auc_score(self.y_true, y_scores)

# Example usage:
# evaluator = ModelEvaluator(y_true, y_pred)
# evaluator.plot_confusion_matrix()
# evaluator.print_classification_report()
# evaluator.plot_roc_curve(y_scores)
# auc = evaluator.calculate_auc(y_scores)
# print(f'AUC: {auc}')
