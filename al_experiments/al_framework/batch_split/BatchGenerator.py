import numpy as np
import pandas as pd

class PeriodicGraphene():
    
    def split_into_subsystem(self, data, label, C, H, O):
        C_element = data["C"] == C
        H_element = data["H"] == H
        O_element = data["O"] == O
        system = data[C_element & H_element & O_element]
        indices = system.index
        label = label[indices]
            
        return system, label, indices
    
    def append_systems(self, sys_list, label_list, X, y):
        
        indices_list = []
        group_list = X.groupby(['C', 'H', 'O']).size().reset_index().rename(columns={0:'count'})
        
        for index, row in group_list.iterrows():
            sys, label, indices = self.split_into_subsystem(X, y, row['C'], row['H'], row['O'])
            sys_list.append(sys)
            label_list.append(label)
            indices_list.append(indices)
            
        return sys_list, label_list, indices_list
    
    
class Nanoparticles():
        
    def split_into_subsystem(self, data, label, T):
        element = data["T"] == T
        system = data[element]
        indices = system.index
        label = label[indices]

        return system, label, indices
  
       
    def append_systems(self, sys_list, label_list, X, y):
        
        indices_list = []
        group_list = X.groupby(['T']).size().reset_index().rename(columns={0:'count'})
        
        for index, row in group_list.iterrows():
            sys, label, indices = self.split_into_subsystem(X, y, row['T'])
            sys_list.append(sys)
            label_list.append(label)
            indices_list.append(indices)
        
        return sys_list, label_list, indices_list
    
    
class SemiConductor():

    def split_into_subsystem(self, data, label, C):
        element = data["components"] == C
        system = data[element]
        indices = system.index
        label = label[indices]
        system = system.drop(['components'], axis=1)

        return system, label, indices
    
    
    def append_systems(self, sys_list, label_list, X, y):
        indices_list = []
        group_list = X.groupby(['components']).size().reset_index().rename(columns={0:'count'})
        
        for index, row in group_list.iterrows():
            sys, label, indices = self.split_into_subsystem(X, y, row['components'])
            sys_list.append(sys)
            label_list.append(label)
            indices_list.append(indices)
        
        return sys_list, label_list, indices_list
