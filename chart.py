# roc curve and auc
import numpy as np
from sklearn import metrics
from matplotlib import pyplot
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

y_true = pd.read_csv("mirai_labels.csv",header=None).to_numpy()
ssa = pd.read_csv("ssa.csv",header=None).to_numpy()
sda = pd.read_csv("sda.csv",header=None).to_numpy()

sda_pred = np.array([1 if i >1 else 0 for i in sda])
ssa_pred = np.array([1 if i >1 else 0 for i in ssa])

fpr1, tpr1, thresholds1 = metrics.roc_curve(y_true, sda_pred, pos_label=1)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_true, ssa_pred, pos_label=1)
print('SDA AUC',metrics.auc(fpr1, tpr1))
print('SSA AUC',metrics.auc(fpr2, tpr2))

sda_eer = brentq(lambda x : 1. - x - interp1d(fpr1, tpr1)(x), 0., 1.)
ssa_eer = brentq(lambda x : 1. - x - interp1d(fpr2, tpr2)(x), 0., 1.)

thresh = interp1d(fpr1, thresholds1)(sda_eer)
print(f"SDA EER {sda_eer}")
thresh = interp1d(fpr2, thresholds2)(ssa_eer)
print(f"SSA EER {ssa_eer}")

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
