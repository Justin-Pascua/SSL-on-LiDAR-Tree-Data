import pandas as pd


def process_labeled_data(paths):
    """
    Returns a pandas dataframe containing all the labeled data samples from
    the Tree LiDAR dataset.
    params:
        paths: a string or list of strings of path(s) to the .csv file(s) containing labeled data
    """
    
    path_list = []
    if type(paths) == list:
        path_list = paths
    else:
        path_list.append(paths)
    
    # merge datasets
    full_labeled_df = pd.DataFrame()
    for path in path_list:
        temp_df = pd.read_csv(path, sep = ';')
        full_labeled_df = pd.concat([full_labeled_df, temp_df])
        
    # Drop unused columns
    full_labeled_df = full_labeled_df.drop(['itot', 'ipcumzq10', 'ipcumzq90', 'Set'], axis = 1)

    # Rename SP3 to Species
    full_labeled_df.rename(columns = {'SP3': "Species"}, inplace = True)
    
    # Clean up species strings
    clean_strings = {'Norway_spruce': 'Norway Spruce',
                     'European_larch': 'European Larch',
                     'Other_broadleaves': 'Other broadleaves',
                     'Silver_fir': 'Silver fir',
                     'Broadleaves': 'Broadleaves',
                     'Green_alder': 'Green alder',
                     'Pines': 'Pines',
                     'Scots_pine': 'Scots Pine'}
    full_labeled_df['Species'] = full_labeled_df['Species'].map(clean_strings)
    
    # Get species names
    species_strings = list(full_labeled_df['Species'].value_counts().index)
    species_string_to_num = {species_strings[i]: i for i in range(8)}
    
    # Convert species strings to numerical labels
    full_labeled_df['Species Number'] = full_labeled_df['Species'].map(species_string_to_num)
    
    return full_labeled_df
    
def process_unlabeled_data(paths):
    """
    Returns a pandas dataframe containing all the unlabeled data samples from
    the Tree LiDAR dataset.
    params:
        paths: a string or list of strings of path(s) to the .csv file(s) containing labeled data
    """
    path_list = []
    if type(paths) == list:
        path_list = paths
    else:
        path_list.append(paths)

    # merge datasets
    full_unlabeled_df = pd.DataFrame()
    for path in path_list:
        temp_df = pd.read_csv(path, sep = ';')
        full_unlabeled_df = pd.concat([full_unlabeled_df, temp_df])
        
    full_unlabeled_df = full_unlabeled_df.drop(['itot', 'ipcumzq10', 'ipcumzq90'], axis = 1)
    
    return full_unlabeled_df