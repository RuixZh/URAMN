import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from models import LogReg
import torch.nn as nn
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def semi_evaluation(X_train, X_test, Y_train, Y_test):
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
    f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
    print('f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))
    n = len(set(Y_test))
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X_test)
    pred_label = kmeans.predict(X_test)
    nmi = metrics.normalized_mutual_info_score(Y_test, pred_label)
    adjscore = metrics.adjusted_rand_score(Y_test, pred_label)
    ami = metrics.adjusted_mutual_info_score(Y_test, pred_label)
    print('NMI: %.5f, ARI: %.5f, AMI: %.5f'%(nmi, adjscore, ami))
    
    
def classification(embeds, labels, isTest=True):
    f1_mi_20, f1_ma_20,f1_mi_40, f1_ma_40,f1_mi_60, f1_ma_60,f1_mi_80, f1_ma_80 = [],[],[],[],[],[],[],[]
    nmis, adjs, amis = [],[],[]
    for rs in [0, 677873687,  299979297, 185929102, 775010785,  4023022221, 802060833,634554150,835264122,353783084]: 

        X_train, X_test, Y_train, Y_test = train_test_split(embeds, labels, test_size=0.6, random_state=rs)
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_train, Y_train)
        Y_pred = lr.predict(X_test)
        f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
        f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
        f1_mi_40.append(f1_micro)
        f1_ma_40.append(f1_macro)

        n = len(set(labels))
        kmeans = KMeans(n_clusters=n, random_state=rs).fit(embeds)
        pred_label = kmeans.predict(embeds)
        nmi = metrics.normalized_mutual_info_score(labels, pred_label)
        adjscore = metrics.adjusted_rand_score(labels, pred_label)
        ami = metrics.adjusted_mutual_info_score(labels, pred_label)
        nmis.append(nmi)
        adjs.append(adjscore)
        amis.append(ami)
    print('training 40%% f1_macro=%f, f1_micro=%f' % (np.mean(f1_ma_40), np.mean(f1_mi_40)))
    print('M2V NMI: %.5f, ARI: %.5f, AMI: %.5f'%(np.mean(nmis), np.mean(adjs), np.mean(amis)))
