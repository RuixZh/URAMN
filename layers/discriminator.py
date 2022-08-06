import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k_bilinear = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0) # c: summary vector, h_pl: positive, h_mi: negative
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k_bilinear(h_pl, c_x), -1) # sc_1 = 1 x nb_nodes
        sc_2 = torch.squeeze(self.f_k_bilinear(h_mi, c_x), -1) # sc_2 = 1 x nb_nodes

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), -1)

        return logits
    
    
    
class Contrast(nn.Module):
    def __init__(self, ones, margin=0.8, p=2.0, tau=1.0):
        super(Contrast, self).__init__()
        self.marginloss = nn.MarginRankingLoss(margin)
        self.ones = ones
        self.tau = tau

    def sim(self, z1, z2):
        dot_numerator=torch.sum(z1*z2,-1)
        return dot_numerator

    def forward(self, z, z_pos, z_neg):
        sim_pos = self.sim(z, z_pos)
        sim_neg = self.sim(z, z_neg)

        loss = self.marginloss(torch.sigmoid(sim_pos), torch.sigmoid(sim_neg), self.ones)
        return loss
    
     
class TripletContrast(nn.Module):
    def __init__(self, margin=0.8, p=2.0, tau=1.0):
        super(TripletContrast, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)

    def forward(self, z, z_pos, z_neg):
        loss = self.triplet_loss(z, z_pos, z_neg)
        return loss   
    
    