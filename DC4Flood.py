import os
import scipy.io as sio
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import _supervised
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.metrics import adjusted_rand_score
os.chdir("D:/DC4Flood/")

from conv_layer import DC4Flood


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)


# =============================================================================
# Clustering Accuracy (CA)
# =============================================================================
def clustering_accuracy(labels_true, labels_pred):
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)
# =============================================================================

# =============================================================================
# Adjust rand index (ARI)
# =============================================================================
def ARI(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    return ari
# =============================================================================

# =============================================================================
# Normalized mutual information (NMI)
# =============================================================================
def NMI(labels_true, labels_pred):
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return nmi
# =============================================================================



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




list_name = ["Bolivia", "Ghana", "India", "Mekong", "Spain", "USA"]
for mnx, count_name in enumerate(list_name):
    print(mnx)
    print(count_name)
    for v in range(5):
        country = count_name
        X_in = sio.loadmat("./Sen1flood/"+str(country)+"_S1_HAND.mat")["data"]
        [m,n,l] = X_in.shape

        print("The dimension of the input image is:"+str(X_in.shape))
        X_in = np.reshape(X_in,(X_in.shape[0]*X_in.shape[1],X_in.shape[2]))
        scaler = MinMaxScaler()
        X_in = scaler.fit_transform(X_in)
        X_in = np.float32(X_in)
        X_in = X_in.reshape((1,m,n,l))
        X_in = X_in.transpose(0,3,1,2)
        tmpt = torch.from_numpy(X_in)


        num_FE = 2


        Y = sio.loadmat("./Sen1flood/"+str(country)+"_S1_HAND.mat")["GT"]
        y_test = Y.reshape((m*n))
        y_test = y_test.astype(int) + 1
        print(np.unique(y_test))

        if torch.cuda.is_available()==True:
            print("Cuda is avaialbe on the device")
            print(f"GPU is: ",torch.cuda.get_device_name(0))
        else:
            print("Cuda is not avaialbe on the device")

        if torch.cuda.device_count() >= 1:
            model = DC4Flood()
            model.cuda()
        else:
            model = DC4Flood()

        model.apply(weights_init)




        # =============================================================================
        # Setting the hyperparameter values, e.g. learning rate, number of iterations, etc.
        # =============================================================================
        LR = 0.01 
        Iter = 200
        thr = 100
        optimizer_model = torch.optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.MSELoss()
        losses = []

        print(model)
        print("Learnable parameters:"+str(count_parameters(model)))



        
        # =============================================================================
        # Training process of DC4Flood
        # =============================================================================



        for i in tqdm(range(Iter)):
            model.train()
            rec_out, code = model(tmpt.cuda())
            loss = loss_func(rec_out, tmpt.reshape((1*l*m*n,1)).cuda())
            losses.append(loss.cpu().item())
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
            print('Iteration: ', i, '| Total loss: %.4f' % loss.data.cpu().numpy())
            if loss.data.cpu().numpy() < thr:
                torch.save(model.state_dict(), './Sen1flood/net_params_model_DC4Flood.pkl')
                thr = loss.data.cpu().numpy()
            torch.cuda.empty_cache()





        model = DC4Flood().cuda()
        model_dict = model.state_dict()
        pretrained_dict = torch.load('./Sen1flood/net_params_model_DC4Flood.pkl')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  
        model.load_state_dict(model_dict)

        x = model.conv1(tmpt.cuda())
        x = model.conv2(x)
        x = model.conv3(x)
        x = model.conv4(x)
        x = model.conv5(x)
        x = model.conv6(x)
        x = model.conv7(x)

        x_dc4flood = x.data.cpu().numpy()
        x_dc4flood = x_dc4flood.reshape((num_FE,m*n))
        EF_dc4flood = x_dc4flood
        x_dc4flood = x_dc4flood.transpose(1,0)

        # =============================================================================
        # Plotting the extracted features (EF)
        # =============================================================================


        fig = plt.figure(figsize= (10, 10))
        for num, img in enumerate(EF_dc4flood):
            ax = fig.add_subplot(EF_dc4flood.shape[0]+2, 2, num+1)
            plt.title(label="Feature No."+str(num+1))
            ax.axis('off')
            ax.imshow(img.reshape((m,n)))

        ax = fig.add_subplot(EF_dc4flood.shape[0]+2, 2, num+2)
        plt.title(label="GT")
        ax.axis('off')
        ax.imshow(y_test.reshape((m,n)))


        # =============================================================================
        # Generating the clustering map using k-means, and plotting the clustering result and loss values
        # =============================================================================


        kmeans = KMeans(n_clusters=2,n_init='auto')
        km_dc4flood = kmeans.fit(x_dc4flood)
        kmap_dc4flood = km_dc4flood.labels_
        kmap_dc4flood = km_dc4flood + 1
        print(np.unique(kmap_dc4flood))
        CA_dc4flood = clustering_accuracy(y_test,kmap_dc4flood)
        NMI_dc4flood = NMI(y_test,kmap_dc4flood)
        ARI_dc4flood = ARI(y_test,kmap_dc4flood)
        SScore_dc4flood = silhouette_score(x_dc4flood,km_dc4flood)

        fig = plt.figure(figsize= (10, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.title(label="Generated Clustering Map")
        plt.imshow(kmap_dc4flood.reshape((m,n)))

        fig = plt.figure(figsize= (10, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.title(label="Loss values")
        plt.plot(losses)



        print("CA_DC4Flood:"+str(CA_dc4flood))
        print("NMI_AE:"+str(NMI_dc4flood))
        print("ARI_AE:"+str(ARI_dc4flood))
        print("SScore_AE:"+str(SScore_dc4flood))


        sio.savemat('./Sen1flood/CSmap_'+str(country)+'_iter'+str(v+1)+'_DC4Flood_S1_HAND.mat', 
                    {'CSmap_dc4flood':kmap_dc4flood.reshape((m,n)),'loss':losses, 'FE_dc4flood':x_dc4flood.reshape((m,n,2)),'Learnable_no':count_parameters(model),
                     'CA_DC4Flood':CA_dc4flood,'NMI_DC4Flood':NMI_dc4flood,'ARI_DC4Flood':ARI_dc4flood,'SScore_dc4flood':SScore_dc4flood})

print("Finished.")




