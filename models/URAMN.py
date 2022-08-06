import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator, Attention, Bernprop, Bernprop2, Contrast, TripletContrast
import numpy as np
np.random.seed(0)
from evaluate import classification, semi_evaluation
from models import LogReg
import pickle as pkl
import torch.nn.functional as F

class URAMN(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        features = self.features.to(self.args.device) 
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        neighbor_adj = [adj_.to(self.args.device) for adj_ in self.neighbor_adj]
        
        
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        cnt_wait = 0; best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        if self.args.isSemi:
            saved_model = 'best_semi_'
        elif self.args.isAttn:
            saved_model = 'best_att_'
        else:
            saved_model = 'best_'
            
        for epoch in range(self.args.nb_epochs):
            xent_loss = None
            model.train()
            optimiser.zero_grad()

            result = model(features, adj, neighbor_adj, self.args.sparse, None, None, None)

            loss = result['loss']

            reg_loss = result['reg_loss']
            loss += self.args.reg_coef * reg_loss

            if self.args.isSemi:
                sup = result['semi']
                semi_loss = xent(sup[self.idx_train], self.train_lbls)
                loss += self.args.sup_coef * semi_loss

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'saved_model/'+saved_model+'{}.pkl'.format(self.args.dataset))

            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break

            loss.backward()
            optimiser.step()


        model.load_state_dict(torch.load('saved_model/'+saved_model+'{}.pkl'.format(self.args.dataset)))

        if self.args.isSemi:
            train_x = model.H[self.idx_train].detach().cpu().numpy()
            test_x = model.H[self.idx_test].detach().cpu().numpy()
            semi_evaluation(train_x, test_x, self.train_y, self.test_y)
            embs = model.H.detach().cpu().numpy()
            np.save("outputs/{}.Z.semi.npy".format(self.args.dataset), embs)
        else:
            print('Results of {}'.format(self.args.dataset))
            embs = model.pos_emb.detach().cpu().numpy()
#             np.save("outputs/{}.H.att.npy".format(self.args.dataset), embs)
            classification(embs, self.truth)
            embs = model.H.detach().cpu().numpy()
#             np.save("outputs/{}.Z.att.npy".format(self.args.dataset), embs)
            classification(embs, self.truth)
        


class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        if self.args.dataset == 'dblp':
            self.fc_1 = nn.Linear(args.ft_size, 2*args.hid_units, bias=args.isBias)
            self.fc_2 = nn.Linear(2*args.hid_units, args.hid_units, bias=args.isBias)
        else:
            self.fc_1 = nn.Linear(args.ft_size, args.hid_units, bias=args.isBias)
            
        self.bern = nn.ModuleList([Bernprop2(2) for _ in range(args.nb_graphs)])
        self.cons = TripletContrast(margin=args.alpha)#args.margin
        self.cons1 = TripletContrast(margin=args.beta)
    
        
        self.H = nn.Parameter(torch.FloatTensor(args.nb_nodes, args.hid_units))
        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        if args.isSemi:
            self.logistic = LogReg(args.hid_units, args.nb_classes).to(args.device)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def forward(self, feature, adj, neighbor_adj, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}
        loss = 0.0
        idx = np.random.permutation(self.args.nb_nodes)
        if self.args.dataset == 'dblp':
            x = self.fc_1(feature)
            x_out = F.relu(x)
            x = self.fc_2(x_out)
        else:
            x = self.fc_1(feature)
            
        fake = self.H[idx]
        for i in range(self.args.nb_graphs):

            bern_h_1, z_pos, z_neg = self.bern[i](x, idx, adj[i], neighbor_adj[i], sparse)
            loss += self.cons(bern_h_1, z_pos, z_neg)
            loss += self.cons1(self.H, bern_h_1, fake)
    
            h_1_all.append(bern_h_1)

        result['loss'] = loss

        # Attention or not
        if self.args.isAttn:
            h_1_all_lst = []; h_2_all_lst = []; c_all_lst = []

            for h_idx in range(self.args.nheads):
                h_1_all_ = self.attn[h_idx](h_1_all)
                h_1_all_lst.append(h_1_all_)

            self.pos_emb = torch.mean(torch.stack(h_1_all_lst), 0)

        else:
            self.pos_emb = torch.mean(torch.stack(h_1_all), 0)   
            
        result['reg_loss'] = ((self.H - self.pos_emb) ** 2).sum()#-((self.H - fake) ** 2).sum()
        # semi-supervised module
        if self.args.isSemi:
            semi = self.logistic(self.H)
            result['semi'] = semi

        return result