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

def evaluate(embeds, idx_train, idx_val, idx_test, labels, device, isTest=True):
    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = [] ##
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.to(device)

        val_accs = []; test_accs = []
        val_micro_f1s = []; test_micro_f1s = []
        val_macro_f1s = []; test_macro_f1s = []
        for iter_ in range(50):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)


        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter]) ###

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

    if isTest:
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                np.std(macro_f1s),
                                                                                                np.mean(micro_f1s),
                                                                                                np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    test_embs = np.array(test_embs.cpu())
    test_lbls = np.array(test_lbls.cpu())

    run_kmeans(test_embs, test_lbls, nb_classes)
    run_similarity_search(test_embs, test_lbls)

def run_similarity_search(test_embs, test_lbls):
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4)))

    st = ','.join(st)
    print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))

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
#     for rs in [0]: 
#         print(rs)
#         X_train, X_test, Y_train, Y_test = train_test_split(embeds, labels, test_size=0.8, random_state=rs)
#         lr = LogisticRegression(max_iter=500)
#         lr.fit(X_train, Y_train)
#         Y_pred = lr.predict(X_test)
#         f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
#         f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
#         f1_mi_20.append(f1_micro)
#         f1_ma_20.append(f1_macro)
#         print('training 20%% f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))

        X_train, X_test, Y_train, Y_test = train_test_split(embeds, labels, test_size=0.6, random_state=rs)
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_train, Y_train)
        Y_pred = lr.predict(X_test)
        f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
        f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
        f1_mi_40.append(f1_micro)
        f1_ma_40.append(f1_macro)
#         print('training 40%% f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))

#         X_train, X_test, Y_train, Y_test = train_test_split(embeds, labels, test_size=0.4, random_state=rs)
#         lr = LogisticRegression(max_iter=500)
#         lr.fit(X_train, Y_train)
#         Y_pred = lr.predict(X_test)
#         f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
#         f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
#         f1_mi_60.append(f1_micro)
#         f1_ma_60.append(f1_macro)
#         print('training 60%% f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))
        
#         X_train, X_test, Y_train, Y_test = train_test_split(embeds, labels, test_size=0.2, random_state=rs)
#         lr = LogisticRegression(max_iter=500)
#         lr.fit(X_train, Y_train)
#         Y_pred = lr.predict(X_test)
#         f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
#         f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
#         f1_mi_80.append(f1_micro)
#         f1_ma_80.append(f1_macro)
#         print('training 80%% f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))

        n = len(set(labels))
        kmeans = KMeans(n_clusters=n, random_state=rs).fit(embeds)
        pred_label = kmeans.predict(embeds)
        nmi = metrics.normalized_mutual_info_score(labels, pred_label)
        adjscore = metrics.adjusted_rand_score(labels, pred_label)
        ami = metrics.adjusted_mutual_info_score(labels, pred_label)
        nmis.append(nmi)
        adjs.append(adjscore)
        amis.append(ami)
#         print('M2V NMI: %.5f, ARI: %.5f, AMI: %.5f'%(nmi, adjscore,ami))
        
        
#     print('training 20%% f1_macro=%f, f1_micro=%f' % (np.mean(f1_ma_20), np.mean(f1_mi_20)))
    print('training 40%% f1_macro=%f, f1_micro=%f' % (np.mean(f1_ma_40), np.mean(f1_mi_40)))
#     print('training 60%% f1_macro=%f, f1_micro=%f' % (np.mean(f1_ma_60), np.mean(f1_mi_60)))
#     print('training 80%% f1_macro=%f, f1_micro=%f' % (np.mean(f1_ma_80), np.mean(f1_mi_80)))

    print('M2V NMI: %.5f, ARI: %.5f, AMI: %.5f'%(np.mean(nmis), np.mean(adjs), np.mean(amis)))

def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    for i in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)

        s1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)

    s1 = sum(NMI_list) / len(NMI_list)

    print('\t[Clustering] NMI: {:.4f}'.format(s1))