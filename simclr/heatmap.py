# Radar Chart for a specific epoch (e.g., 80 epochs)
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data from Table 4.6
epochs = [20, 40, 60, 80, 100]
simclr_data = {
    'Accuracy': [85.07, 84.75, 85.11, 84.29, 84.93],
    'Sensitivity': [81.76, 81.35, 82.17, 85.42, 82.25],
    'Precision': [90.78, 90.57, 90.49, 86.48, 90.10],
    'F1-score': [86.03, 85.71, 86.13, 85.95, 85.99],
    'AUC': [92.20, 91.20, 92.00, 91.80, 92.30]
}
moco_data = {
    'Accuracy': [83.92, 84.33, 71.74, 85.52, 79.94],
    'Sensitivity': [87.46, 81.76, 98.78, 88.11, 95.93],
    'Precision': [84.50, 89.48, 66.83, 86.42, 75.22],
    'F1-score': [85.95, 85.45, 79.72, 87.26, 84.32],
    'AUC': [92.60, 92.50, 92.40, 93.50, 93.00]
}
# Heatmap for all metrics and epochs
# Combine data into a single DataFrame for heatmap
data = []
for e in epochs:
    for model in ['SimCLR v2', 'MoCo v2']:
        d = simclr_data if model == 'SimCLR v2' else moco_data
        for metric in metrics:
            data.append([e, model, metric, d[metric][epochs.index(e)]])

df = pd.DataFrame(data, columns=['Epoch', 'Model', 'Metric', 'Value'])

# Pivot for heatmap (average across models or separate)
pivot_simclr = df[df['Model'] == 'SimCLR v2'].pivot(index='Epoch', columns='Metric', values='Value')
pivot_moco = df[df['Model'] == 'MoCo v2'].pivot(index='Epoch', columns='Metric', values='Value')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(pivot_simclr, annot=True, cmap='YlGnBu', ax=ax1, cbar_kws={'label': 'Percentage (%)'})
ax1.set_title('SimCLR v2 Performance Heatmap')
sns.heatmap(pivot_moco, annot=True, cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'Percentage (%)'})
ax2.set_title('MoCo v2 Performance Heatmap')
plt.tight_layout()
plt.show()

