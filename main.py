from train import full_train
from utils_imt import load_data
import pandas as pd
import numpy as np
from train import path1, path2



def attribution_column(path1,output):
 #Taking in argument the output of the model trained with the feature of our problem, the function will return the ordenated list of the final attributions

  df=pd.read_csv(path1)
  Cl=list(df['Classes'])
  idx_c=[]
  idx_d=[]
  print(type(Cl[0]))
  for i in range(len(Cl)):
    if Cl[i] == 0:
      idx_c.append(i)
    else :
      idx_d.append(i)
 
  O=output.cpu().detach().numpy()
  Oa=np.abs(O)

  M=np.zeros(len(O))
  for i in idx_c:
    liste=list(O[i])
    max_value = max(liste)
    
    M[i]=liste.index(max_value)+1

  for i in idx_d:
    M[i]=Cl[i]
  dff=df.copy()
  return M, dff

model=full_train() 
adj, features, labels, idx_train, idx_f = load_data(path1,path2)
model.eval()
output = model(features, adj)

list_attr,df_final=attribution_column(path1,output)

df_final['Attribute']=list_attr

#Download the csv with attributions

df_final.to_csv('Att_finale.csv')





