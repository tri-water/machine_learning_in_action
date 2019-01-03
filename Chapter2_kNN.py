#%%
import kNN
import matplotlib.pyplot as plt
%matplotlib inline
#%%
group, labels = kNN.create_dataset()
#%%
group

#%%
labels

#%%
plt.scatter(group[:, 0], group[:, 1])
for i in range(len(labels)):
    plt.text(group[i, 0] + 0.05, group[i, 1], labels[i])
plt.xlim([-0.2, 1.2])

#%%
kNN.classify([0, 0], group, labels, 3)
#%%
loc = r'C:\Users\gao_x\OneDrive\tensorflow_study\machine_learning_in_action\data'

kNN.file2matrix()