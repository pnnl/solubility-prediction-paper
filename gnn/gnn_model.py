import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import EdgeConv
from torch.nn import Linear

params = {'a1': 0, 'a2': 2, 'a3': 1, 'a4': 2, 'bs': 1, 'd1': 0.015105134306121593, 'd2': 0.3431295462686682, \
      'd3': 0.602688496976768, 'd4': 0.9532038077650021, 'e1': 256.0, 'eact1': 0, 'edo1': 0.4813038851902818,\
      'f1': 256.0, 'f2': 256.0, 'f3': 160.0, 'f4': 24.0, 'g1': 256.0, 'g2': 320.0, 'g21': 448.0,\
      'g22': 512.0, 'gact1': 2, 'gact2': 2, 'gact21': 2, 'gact22': 0, 'gact31': 2, 'gact32': 1, 'gact33': 1,\
      'gdo1': 0.9444250299450242, 'gdo2': 0.8341272742321129, 'gdo21': 0.7675340644596443,\
      'gdo22': 0.21498171859119775, 'gdo31': 0.8236003195596049, 'gdo32': 0.6040220843354102,\
      'gdo33': 0.21007469160431758, 'lr': 0, 'nfc': 0, 'ngl': 1, 'opt': 0}
act = {0: torch.nn.ReLU(), 1:torch.nn.SELU(), 2:torch.nn.Sigmoid()}

class GNN(torch.nn.Module):
    
    def __init__(self, n_features):
        super(GNN, self).__init__()
        self.n_features = n_features
        self.gcn1 = GCNConv(self.n_features, int(params['g1']), cached=False) 
        self.gcn2 = GCNConv( int(params['g1']), int(params['g2']), cached=False)
        self.gcn21 = GCNConv( int(params['g2']), int(params['g21']), cached=False)
        self.gcn22 = GCNConv( int(params['g21']), int(params['g22']), cached=False)

        self.gcn31 = GCNConv(int(params['g2']), int(params['e1']), cached=False)
        self.gcn32 = GCNConv(int(params['g21']), int(params['e1']), cached=False)
        self.gcn33 = GCNConv(int(params['g22']), int(params['e1']), cached=False)

        self.gdo1 = nn.Dropout(p = params['gdo1'] )
        self.gdo2 = nn.Dropout(p = params['gdo2'] )
        self.gdo31 = nn.Dropout(p = params['gdo31'] )
        self.gdo21 = nn.Dropout(p = params['gdo21'] )
        self.gdo32 = nn.Dropout(p = params['gdo32'] )
        self.gdo22 = nn.Dropout(p = params['gdo22'] )
        self.gdo33 = nn.Dropout(p = params['gdo33'] )

        self.gact1 = act[params['gact1'] ]
        self.gact2 = act[params['gact2'] ]
        self.gact31 = act[params['gact31']] 
        self.gact21 = act[params['gact21'] ]
        self.gact32 = act[params['gact32'] ]
        self.gact22 = act[params['gact22'] ]
        self.gact33 = act[params['gact33'] ]

        self.ecn1 = EdgeConv(nn = nn.Sequential(nn.Linear(n_features*2, int(params['e1']) ),
                                          nn.ReLU(), 
                                          nn.Linear( int(params['e1']) , int(params['f1'])  ),))

        self.edo1 = nn.Dropout(p = params['edo1'] )
        self.eact1 = act[params['eact1'] ]


        self.fc1 = Linear( int(params['e1'])+ int(params['f1']), int(params['f1']))
        self.dropout1 = nn.Dropout(p = params['d1'] )
        self.act1 = act[params['a1']]

        self.fc2 = Linear(int(params['f1']), int(params['f2']))
        self.dropout2 = nn.Dropout(p = params['d2'] )
        self.act2 = act[params['a2']]

        self.fc3 = Linear(int(params['f2']), int(params['f3']))
        self.dropout3 = nn.Dropout(p = params['d3'] )
        self.act3 = act[params['a3']]

        self.fc4 = Linear(int(params['f3']), int(params['f4']))
        self.dropout4 = nn.Dropout(p = params['d4'] )
        self.act4 = act[params['a4']]

        self.out2 = Linear(int(params['f2']), 1)
        self.out3 = Linear(int(params['f3']), 1)
        self.out4 = Linear(int(params['f4']), 1)


    def forward(self, data):
        node_x, edge_x, edge_index = data.x, data.edge_attr, data.edge_index

        x1 = self.gdo1(self.gact1( self.gcn1( node_x, edge_index ) ) )
        x1 = self.gdo2(self.gact2(self.gcn2(x1, edge_index)) )
        x1 = self.gdo21(self.gact21(self.gcn21(x1, edge_index)) )
        x1 = self.gdo32(self.gact32(self.gcn32(x1, edge_index)) )
        
        x2 = self.edo1(self.eact1(self.ecn1(node_x, edge_index)) )
        x3 = torch.cat((x1,x2), 1)
        x3 = global_add_pool(x3, data.batch)
        x3 = self.dropout1(self.act1(self.fc1( x3 )))
        x3 = self.dropout2(self.act2(self.fc2( x3 )))
        x3 = self.out2(x3)
        return x3
