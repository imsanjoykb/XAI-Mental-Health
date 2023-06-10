import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import numpy as np
import pandas as pd

class Network2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network2, self).__init__()
        self.layer1_size = hidden_size
        self.layer2_size = output_size
        self.total_activations = hidden_size + output_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward_layer1( self, x ):
        # hidden layer
        y = torch.relu( self.linear1(x) )
        return y

    def forward_layer2( self, x ):
        # output layer
        y = torch.sigmoid( self.linear2(x) )        
        return y 
    
    def forward( self, x ):
        x = self.forward_layer1( x )
        y = self.forward_layer2( x )
        return( y )

    def print_layer_names( self ):
        for param_tensor in self.state_dict():
            double_x = self.state_dict()[param_tensor].numpy()            
            print( param_tensor )
    
    def get_activations( self, this_input ):
        # pass a torch.tensor as input, returns a row (numpy array) of activations
        # so you need to know each layer's dimensions to interpret which columns correspond to which cols
        self.eval()
        with torch.no_grad():
            first_layer = self.forward_layer1( this_input )
            second_layer = self.forward_layer2( first_layer )

        first_layer = first_layer.numpy()
        second_layer = second_layer.numpy()

        return( np.concatenate( [first_layer, second_layer] ) )


class Network1(nn.Module):


    def __init__(self, input_size, output_size):
        super(Network1, self).__init__()
        self.layer1_size = input_size
        self.layer2_size = output_size
        self.total_activations = 2  * output_size # for this model, the only activations are the input and output from layer 2

        self.linear = nn.Linear(input_size, output_size)
    
    def forward_layer1( self, x ):
        y = self.linear(x)
        return y

    def forward_layer2( self, x ):
        y = torch.sigmoid(x)
        return y 
    
    def forward( self, x ):
        y = self.forward_layer1( x )
        y = self.forward_layer2( y )
        return( y )

    def print_layer_names( self ):
        for param_tensor in self.state_dict():
            double_x = self.state_dict()[param_tensor].numpy()            
            print( param_tensor )
    
    def get_activations( self, this_input ):
        # pass a torch.tensor as input, returns a row (numpy array) of activations
        # so you need to know each layer's dimensions to interpret which columns correspond to which cols
        self.eval()
        with torch.no_grad():
            first_layer = self.forward_layer1( this_input )
            second_layer = self.forward_layer2( first_layer )

        first_layer = first_layer.numpy()
        second_layer = second_layer.numpy()

        return( np.concatenate( [first_layer, second_layer] ) )


def save_network(network, ident_label, epoch_label):
    save_filename = ident_label + '-net-state_epoch' + str(epoch_label) + '.pth'
    save_path = os.path.join('../net-states', save_filename)
    torch.save(network.state_dict(), save_path)


def load_network(network, fname):
    load_path = os.path.join('../net-states', fname)
    network.load_state_dict( torch.load(load_path) )

def parameters_to_csv( model, this_label ):
    count = 0
    for w in list(model.parameters()):
        count += 1
        this_param = w.detach().numpy()
        save_filename = this_label + '-net-params-' + str(count) + '.csv'
        save_path = os.path.join('../out-data', save_filename)
        df = pd.DataFrame( this_param )
        df.to_csv(save_path, header = False, index = False)

