import os
import pandas as pd

root_dir = 'data/'
aline_dir = root_dir + 'aline/'
bline_dir = root_dir + 'bline/'

files = []
labels = []

for filename in os.listdir(aline_dir):
    files.append(os.path.join(aline_dir, filename))
    labels.append(0)

for filename in os.listdir(bline_dir):
    files.append(os.path.join(bline_dir, filename))
    labels.append(1)

df = pd.DataFrame({'image': files, 'label': labels})
df.to_csv('data.csv', index=False)