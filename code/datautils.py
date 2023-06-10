# prepare data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class GenDataset( Dataset ):

  def __init__( self, filepath, pred_vars, target_var, trainflag = True ):
    df_csv = pd.read_csv( filepath, index_col=0)

    if( trainflag == True ):
      # extract and build the training data ...
      df = df_csv.loc[ df_csv["train"] == 1, : ]
    else:
      # extract and build test data ...
      df = df_csv.loc[ df_csv["train"] == 0, : ]
    
    x = df.loc[:, pred_vars ].values   # predictor variables 
    y = df.loc[:, target_var].values   # target

    self.x = torch.tensor( x, dtype = torch.float32 )
    self.y = torch.tensor( y, dtype = torch.float32 )

  def __len__(self):
    return len( self.y )

  def __getitem__( self, idx ):
    return self.x[idx], self.y[idx]

