# roc curve and auc
import numpy as np
from sklearn import metrics
from matplotlib import pyplot
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

y_true = pd.read_csv("mirai_labels.csv",header=None).to_numpy()
ssa = pd.read_csv("mirai_ssae.csv",header=None).to_numpy()
sda = pd.read_csv("mirai_sdae.csv",header=None).to_numpy()

sda_pred = np.array([1 if i >1 else 0 for i in sda])
ssa_pred = np.array([1 if i >1 else 0 for i in ssa])

fpr1, tpr1, thresholds1 = metrics.roc_curve(y_true, sda_pred, pos_label=1)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_true, ssa_pred, pos_label=1)
print('SDA',metrics.auc(fpr1, tpr1))
print('SSA',metrics.auc(fpr2, tpr2))

sda_eer = brentq(lambda x : 1. - x - interp1d(fpr1, tpr1)(x), 0., 1.)
ssa_eer = brentq(lambda x : 1. - x - interp1d(fpr2, tpr2)(x), 0., 1.)

thresh = interp1d(fpr1, thresholds1)(sda_eer)
print(f"SDA EER {sda_eer}")
thresh = interp1d(fpr2, thresholds2)(ssa_eer)
print(f"SSA EER {ssa_eer}")

print('Precision sda: %.8f' % precision_score(y_true, sda_pred))
print('Precision ssa: %.8f' % precision_score(y_true, ssa_pred))
print('Accuracy sda: %.8f' % accuracy_score(y_true, sda_pred))
print('Accuracy ssa: %.8f' % accuracy_score(y_true, ssa_pred))
print('Recall sda: %.8f' % recall_score(y_true, sda_pred))
print('Recall ssa: %.8f' % recall_score(y_true, ssa_pred))
print('F1 score sda: %.8f' % f1_score(y_true, sda_pred))
print('F1 score ssa: %.8f' % f1_score(y_true, ssa_pred))

conf_matrix = confusion_matrix(y_true, sda_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

conf_matrix = confusion_matrix(y_true, ssa_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()



plt.figure()
lw = 2
plt.plot(fpr1, tpr1, color='green', lw=lw, label='SDA',linestyle='-.')
plt.plot(fpr2, tpr2, color='orange', lw=lw, label='SSA',linestyle=':')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
