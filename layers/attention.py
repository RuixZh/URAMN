import torch.nn as nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.A = nn.ModuleList([nn.Linear(args.hid_units, 1) for _ in range(args.nb_graphs)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.args.nb_graphs):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, feat_pos):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)

        return feat_pos


    def attn_feature(self, features):
        features_attn = []
        for i in range(self.args.nb_graphs):
            features_attn.append((self.A[i](features[i])))
        features_attn = F.softmax(torch.cat(features_attn, -1), -1)
        features = torch.cat(features,0)
        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(self.args.nb_graphs, self.args.nb_nodes, self.args.hid_units).sum(0)
        return features, features_attn

    def attn_summary(self, features):
        features_attn = []
        for i in range(self.args.nb_graphs):
            features_attn.append((self.A[i](features[i])))
        features_attn = F.softmax(torch.cat(features_attn), dim=-1)
        features = torch.cat(features, 0)
        features_attn_expanded = features_attn.expand_as(features)
        features = (features * features_attn_expanded).sum(0)

        return features, features_attn
    
class SemanticAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(SemanticAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()
#input (PN)*F 
    def forward(self, inputs, P):
        h = torch.mm(inputs, self.W)
        #h=(PN)*F'
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0],1))
        #h_prime=(PN)*F'
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P,-1)       
        #semantic_attentions = P*N
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1,keepdim=True)
        #semantic_attentions = P*1
        semantic_attentions = F.softmax(semantic_attentions, dim=0)
#         print(semantic_attentions.item())
        semantic_attentions = semantic_attentions.view(P,1,1)
        semantic_attentions = semantic_attentions.repeat(1,N,self.in_features)
#        print(semantic_attentions)
        #input_embedding = P*N*F
        input_embedding = inputs.view(P,N,self.in_features)
        
        #h_embedding = N*F
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()
        
        return h_embedding

    
class Integrate(nn.Module):
    def __init__(self, args):
        super(Integrate, self).__init__()
        self.args = args
        self.temp = nn.Parameter(torch.Tensor(self.args.nb_graphs))
        self.temp.data.fill_(1)

    def forward(self, adj):
        Temp = F.softmax(self.temp, dim=1)        
        summary = (Temp[0] * adj[0].to_dense()).to_sparse()
        for i in range(1, self.args.nb_graphs):
            summary += (Temp[i] * adj[i].to_dense()).to_sparse()
        return summary
