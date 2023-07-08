from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import torch
from scipy import stats
a = torch.tensor([0,1,2,3,4,5,6,7,8,9])*1.
quantile = torch.quantile(a,0.9,interpolation='lower')
print(quantile)