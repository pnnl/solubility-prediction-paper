import schnetpack
import schnetpack.atomistic as atm
import torch
import schnetpack.representation as rep
import torch.nn as nn


class MolecularOutput(atm.Atomwise):
    def __init__(self, property_name, n_in=128, n_out=1, aggregation_mode='sum',
                 n_layers=2, n_neurons=None,
                 activation=schnetpack.nn.activations.shifted_softplus,
                 outnet=None):
        super(MolecularOutput, self).__init__(n_in, n_out)
        self.property_name = property_name
        self.n_layers = n_layers
        self.create_graph = False
        
        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack.nn.base.GetItem('representation'),
                schnetpack.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation)
            )
        else:
            self.out_net = outnet
        
        if aggregation_mode == 'sum':
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == 'avg':
            self.atom_pool = schnetpack.nn.base.Aggregate(axis=1, mean=True)
            
    def forward(self, inputs):
        r"""
        predicts molecular property
        """
        atom_mask = inputs[schnetpack.Properties.atom_mask]

        yi = self.out_net(inputs)
        y = self.atom_pool(yi, atom_mask)

        result = {self.property_name: y}

        return result
    
    
    
output_key = "sol"
target_key = "sol"

space = {'n_atom_basis': [64, 128],
         'n_filters':  [64,128],
         'n_interactions':  [3,4,5,6,7,8,9,10],
         'n_layers':  [1,2,3,4],
         'aggregation_mode':  ['sum', 'avg']
        }

args = {'aggregation_mode': 1, 'n_atom_basis': 0, 'n_filters': 1, 'n_interactions': 2, 'n_layers': 3}

reps = rep.SchNet(n_atom_basis =  space['n_atom_basis'][ args['n_atom_basis']  ], 
                  n_filters =   space['n_filters'][ args['n_filters'] ]  , 
                  n_interactions = space['n_interactions'][ args['n_interactions']  ] )

output = MolecularOutput(property_name='sol', n_in=  space['n_atom_basis'][ args['n_atom_basis'] ],  
                         aggregation_mode = space['aggregation_mode'][ args['aggregation_mode'] ], 
                         n_layers = space['n_layers'][ args['n_layers'] ]
                        )


model = atm.AtomisticModel(reps, output)
 
    