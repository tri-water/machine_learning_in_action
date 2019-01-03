#%%
import kNN
import matplotlib.pyplot as plt
import numpy as np
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
# Example: Improve matches from a dating site with kNN
loc = r'C:\Users\gao_x\OneDrive\tensorflow_study\machine_learning_in_action\data'
dating_data_mat, dating_labels = kNN.file2matrix(loc, 'datingTestSet.txt')

#%%
dating_data_mat

#%%
dating_labels[:20]

#%%
# Analyze the data
# Create scatter plots with Matplotlib
dating_labels_id = np.zeros(len(dating_labels))
dating_labels_id[dating_labels=='didntLike'] = 1
dating_labels_id[dating_labels=='smallDoses'] = 2
dating_labels_id[dating_labels=='largeDoses'] = 3
plt.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2],
            15.0*dating_labels_id, 15.0*dating_labels_id)
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed per Week')
plt.show()
#%%
plt.scatter(dating_data_mat[:, 1], dating_data_mat[:, 0],
            15.0*dating_labels_id, 15.0*dating_labels_id)
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Frequent Flyier Miles Earned Per Year')
plt.show()

#%%
norm_mat, ranges, min_vals = kNN.auto_norm(dating_data_mat)

#%%
kNN.dating_class_test(loc, 'datingTestSet.txt')

#%%
# Example: a handwriting recognition system
testing_loc = r'C:\Users\gao_x\OneDrive\tensorflow_study\machine_learning_in_action\data\digits\testDigits'
testing_vector = kNN.img2vector(testing_loc, '0_13.txt')
testing_vector[:31]

#%%
testing_vector[32:63]

#%%
training_loc = r'C:\Users\gao_x\OneDrive\tensorflow_study\machine_learning_in_action\data\digits\trainingDigits'
kNN.hand_writing_class_test(testing_loc, training_loc)