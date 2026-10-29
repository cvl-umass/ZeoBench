import os
import csv
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class ZeoDataset(Dataset):
    sample_id_key = 'zeolite'
    target_key = 'kH_C18'

    def __init__(self, root_dir, target='C18', is_train=True, is_val=False, full=True, transform=None, custom_train_file=None, use_handcrafted=False):
        '''
        Zeolite distance grids Dataset.
        Args: 
            root_dir (string): Root directory of dataset
            target (string): Target molecule for regression (e.g., 'C18') or 'synthesizability'
            is_train (bool): Indicates if this is the training set
            is_val (bool): Indicates if this is the validation set
            full (bool): If True, uses all available data not in val/test. If False, respects subset files.
            transform (callable, optional): Transform to be applied on a sample.
            custom_train_file (string, optional): Path to a specific subset txt file.
            use_handcrafted (bool): If True, returns tabular physical features instead of 3D grids.
        '''
        self.root_dir = root_dir
        self.distance_folder = 'distance-grids-h5'
        self.is_train = is_train
        self.full = full
        self.transform = transform
        self.use_handcrafted = use_handcrafted

        dist_folder = 'distance-grids-h5'              
        annotations_folder = 'C18-adsorption'       
        annotations_file = 'each-zeolite-info.csv'  
        
        if custom_train_file:
            train_path = custom_train_file
        else:
            train_path = os.path.join(root_dir, annotations_folder, 'train_set.txt')
            
        val_file = 'val_set.txt'  
        test_file = 'test_set.txt'

        mol_to_idx = {
            'C18': 2, 'C24': 3, 'C30': 4,
            '2C17': 5, '4C17': 6, '22C16': 7
        }
    
        # Retrieve filenames and labels
        files = {}
        with open(os.path.join(root_dir, annotations_folder, annotations_file), 'r') as csvfile:
            for line in csv.reader(csvfile):
                if line[0] == 'zeolite': continue   
                
                # IZA-SC: Database of approved zeolite structures
                # PCOD: Predicted crystallography open database
                db_set = "IZASC" if line[1] == "IZA" else "PCOD" 

                if target == 'synthesizability':
                    label = 1.0 if db_set == "IZASC" else 0.0
                else:
                    raw_val = line[mol_to_idx[target]]
                    if raw_val == '0':
                        label = float('nan') 
                        if not full:
                            continue
                    else:
                        label = np.log(float(raw_val))
                        
                if not os.path.exists(os.path.join(root_dir, dist_folder, db_set, line[0]+'.h5py')): 
                    continue                                    

                metadict = {
                    'zeolite': line[0],
                    'set': db_set,
                    'kH_C18': label
                }
                
                if not full:
                    metadict.update({
                        'largest_free_sphere': float(line[36]),
                        'largest_inc_sphere': float(line[35]),
                        'surface_area_m2_g': float(line[39]),
                        'pore_volume_cm3_g': float(line[42]),
                        'T_density_g_cm3': float(line[40])
                    })
                files[line[0]] = metadict
            
        # Split dataset (train/val/test) based on csv/txt files
        self.files = []
        if not full:
            if is_train:
                with open(train_path, 'r') as csvfile:
                    for line in csv.reader(csvfile):
                        if line[0] == 'zeolite': continue    
                        if line[0] in files:
                            self.files.append(files[line[0]])
            elif is_val and not is_train:           
                with open(os.path.join(root_dir, annotations_folder, val_file), 'r') as csvfile:
                    for line in csv.reader(csvfile):
                        if line[0] == 'zeolite': continue    
                        if line[0] in files:
                            self.files.append(files[line[0]]) 
            elif not is_train and not is_val:       
                with open(os.path.join(root_dir, annotations_folder, test_file), 'r') as csvfile:
                    for line in csv.reader(csvfile):
                        if line[0] == 'zeolite': continue    
                        if line[0] in files:
                            self.files.append(files[line[0]])
        else:
            with open(os.path.join(root_dir, annotations_folder, val_file), 'r') as csvfile:
                for line in csv.reader(csvfile):
                    if line[0] == 'zeolite': continue    
                    if line[0] in files:
                        del files[line[0]]
            with open(os.path.join(root_dir, annotations_folder, test_file), 'r') as csvfile:
                for line in csv.reader(csvfile):
                    if line[0] == 'zeolite': continue    
                    if line[0] in files:
                        del files[line[0]]
            self.files = list(files.values())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.use_handcrafted:
            feat = torch.tensor([
                self.files[idx]['largest_free_sphere'], 
                self.files[idx]['largest_inc_sphere'], 
                self.files[idx]['surface_area_m2_g'], 
                self.files[idx]['pore_volume_cm3_g'], 
                self.files[idx]['T_density_g_cm3']
            ])
            return feat, self.files[idx].get('kH_C18', float('nan'))
            
        distance_name = os.path.join(self.root_dir, self.distance_folder, self.files[idx]['set'], self.files[idx]['zeolite']+'.h5py')
        
        with h5py.File(distance_name, 'r') as f:
            grid = torch.Tensor(np.array(f['distancegrids'])).unsqueeze(0)
            _, nx, ny, nz = grid.shape
            bounds = torch.Tensor([nx, ny, nz])
            
            xvec = f['distancegrids'].attrs['xvec']
            yvec = f['distancegrids'].attrs['yvec']
            zvec = f['distancegrids'].attrs['zvec']
        
        raw_data = (grid, xvec, yvec, zvec, bounds, nx, ny, nz)
        label = self.files[idx].get('kH_C18', float('nan'))
        
        if self.transform:
            augmented_grid = self.transform(raw_data)
            return augmented_grid, label
        
        return grid, label
