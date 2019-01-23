import os
import numpy as np
import random
import matplotlib.pyplot as plt


def load_dataset(file_name):
    cwd = os.getcwd()
    with open(cwd + r'/data/' + file_name) as file:
        line_list = file.readlines()
        line_list = [line.strip().split('\t') for line in line_list]
        line_list = np.mat(line_list, dtype=float)
        data_mat = line_list[:, :-1]
        label_mat = line_list[:, -1]
    return data_mat, label_mat


def select_j_rand(i, m):
    """Select a random value, which is not equal to input i

    Params:
    =======
    i: the index of our first alpha
    m: the total number of alphas
    """
    
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(a_j, h, l):
    """Clip alpha values that are greater than h or less than l"""

    if a_j > h:
        a_j = h
    if a_j < l:
        a_j = l
    return a_j


def smoSimple(dataMatrix, labelMat, C, toler, maxIter):
    b = 0
    m,n = np.shape(dataMatrix)    
    alphas = np.mat(np.zeros((m,1)))    
    iter = 0    
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T* \
                       (dataMatrix*dataMatrix[i,:].T)) + b            
            Ei = fXi - float(labelMat[i])            
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
               ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                    j = select_j_rand(i,m)
                    fXj = float(np.multiply(alphas,labelMat).T*\
                               (dataMatrix*dataMatrix[j,:].T)) + b       
                    Ej = fXj - float(labelMat[j])                
                    alphaIold = alphas[i].copy() 
                    alphaJold = alphas[j].copy()               
                    if (labelMat[i] != labelMat[j]):           
                        L = max(0, alphas[j] - alphas[i])    
                        H = min(C, C + alphas[j] - alphas[i])   
                    else:                          
                        L = max(0, alphas[j] + alphas[i] - C)     
                        H = min(C, alphas[j] + alphas[i])        
                    if L==H: 
                        print("L==H")
                        continue                
                    eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - \
                            dataMatrix[i,:]*dataMatrix[i,:].T - \
                            dataMatrix[j,:]*dataMatrix[j,:].T    
                    if eta >= 0: 
                        print("eta>=0")
                        continue                
                    alphas[j] -= labelMat[j]*(Ei - Ej)/eta                
                    alphas[j] = clip_alpha(alphas[j],H,L)                
                    if (abs(alphas[j] - alphaJold) < 0.00001): 
                        print("j not moving enough")
                        continue                
                    alphas[i] += labelMat[j]*labelMat[i]*\
                                 (alphaJold - alphas[j])          
                    b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*\
                         dataMatrix[i,:]*dataMatrix[i,:].T - \
                         labelMat[j]*(alphas[j]-alphaJold)*\
                         dataMatrix[i,:]*dataMatrix[j,:].T         
                    b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*\
                        dataMatrix[i,:]*dataMatrix[j,:].T - \
                        labelMat[j]*(alphas[j]-alphaJold)*\
                        dataMatrix[j,:]*dataMatrix[j,:].T    
                    if (0 < alphas[i]) and (C > alphas[i]): 
                        b = b1                      
                    elif (0 < alphas[j]) and (C > alphas[j]): 
                        b = b2                    
                    else: 
                        b = (b1 + b2)/2.0                    
                    alphaPairsChanged += 1                
                    print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
    if (alphaPairsChanged == 0): 
        iter += 1       
    else: 
        iter = 0        
    print("iteration number: %d" % iter)
    return b, alphas


def smo_simple(data_mat, label_mat, C, tol, max_iter):
    b = 0
    m, n = data_mat.shape
    alphas = np.mat(np.zeros((m, 1)))

    iter = 0
    while iter < max_iter:
        # Record if the attempt to optimize any alpha worked
        alpha_parirs_changed = 0
        for i in range(m):
            # Prediction of the class
            fX_i = float(np.multiply(alphas, label_mat).T\
                   * (data_mat*data_mat[i, :].T)) + b
            # the error of the prediction from the real class
            # If this error is large, then the alpha corresponding to
            # this data instance can be optimized
            E_i = fX_i - float(label_mat[i]) 

            if (label_mat[i]*E_i < -tol) and (alphas[i] < C)\
                or ((label_mat[i]*E_i > tol) and (alphas[i] > 0)):
                # If the absolute value of the margin is tested
                # We also check to see that the alpha isn't equal to
                # 0 or C

                # Randomly select a second alpha
                j = select_j_rand(i, m)
            
                # Calculate the error for this alpha
                fX_j = float(np.multiply(alphas, label_mat).T\
                       * (data_mat*data_mat[j, :].T)) + b
                E_j = fX_j - float(label_mat[j])
                
                # Record the two alphas
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()

                # L and H are used to clamp alpha[j]
                if (label_mat[i] != label_mat[j]):
                    L = max(0, float(alphas[j] - alphas[i]))
                    H = min(C, float(C + alphas[j] - alphas[i]))
                else:
                    L = max(0, float(alphas[j] + alphas[i] - C))
                    H = min(C, float(alphas[j] + alphas[i]))
                if L == H:
                    # If L and H are equal, you cannot do anything
                    print("L == H")
                    continue
                
                # eta is the optimal amount to change alphas[j]
                eta = float(2. * data_mat[i, :] * data_mat[j, :].T\
                      - data_mat[i, :] * data_mat[i, :].T\
                      - data_mat[j, :] * data_mat[j, :].T)
                if eta >= 0:
                    print("eta >= 0")
                    continue

                alphas[j] -= label_mat[j]*(E_i - E_j) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)

                if (abs(alphas[j] - alpha_j_old) < 0.00001):
                    print("j not moving enough")
                    continue

                # alpha[i] is changed by the same amount as alphas[j]
                # but in the opposite direction
                alphas[i] += label_mat[j] * label_mat[i]\
                             * (alpha_j_old - alphas[j])
                b1 = b - E_i - label_mat[i] * (alphas[i] - alpha_i_old)\
                     * data_mat[i, :] * data_mat[i, :].T\
                     - label_mat[j] * (alphas[j] - alpha_j_old)\
                     * data_mat[i, :] * data_mat[j, :].T
                b2 = b - E_j - label_mat[i]*(alphas[i] - alpha_i_old)\
                     * data_mat[i, :] * data_mat[j, :].T\
                     - label_mat[j]*(alphas[j] - alpha_j_old)\
                     * data_mat[j, :] * data_mat[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.
                alpha_parirs_changed += 1
                print("iter: {0} i: {1}, pairs changed {2}".format(
                    iter, i, alpha_parirs_changed
                ))
        if (alpha_parirs_changed == 0):
                iter += 1
        else:
                iter = 0
        print("iteration number: {0}".format(iter))
    return b, alphas
        



if __name__ == '__main__':
    data_mat, label_mat = load_dataset('testSetCh06.txt')
    label_mat[label_mat==0] = -1
    b, alphas = smo_simple(data_mat, label_mat, 0.6, 0.001, 40)
    print((label_mat.A1>0).shape)
    plt.scatter(data_mat[label_mat.A1>0, 0].A1, data_mat[label_mat.A1>0, 1].A1,
                c='r', marker='s')
    plt.scatter(data_mat[label_mat.A1<0, 0].A1, data_mat[label_mat.A1<0, 1].A1,
                c='g', marker='o')

    for i, selected in enumerate(alphas>0):
        if selected:
            plt.scatter(data_mat[i,0], data_mat[i,1], edgecolors='b', facecolors='None', s=300,
                        marker='o')
    plt.show()






