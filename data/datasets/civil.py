import numpy as np
import pandas as pd

RACE_LIST = ['asian', 'black']
RACE_ETC = ['white', 'latino']
DATAPATH = '/data/share/FairLLM/civil/'
ACC_BAR = 0.4
    

def preprocess_civil_roberta_data(seed): 
    np.random.seed(seed)

    features = np.load(DATAPATH + 'roberta_base_train_features.npy')
    df = pd.read_csv(DATAPATH + 'all_sensitive_data.csv')
    
    ind_0 = np.where((df[RACE_LIST[0]] > 0.0) & (df[RACE_LIST[1]] == 0.0) & ((df[RACE_ETC] != 0).sum(axis = 1) < 1))[0]
    ind_1 = np.where((df[RACE_LIST[1]] > 0.0) & (df[RACE_LIST[0]] == 0.0) & ((df[RACE_ETC] != 0).sum(axis = 1) < 1))[0]
    
    features_processed = np.vstack([
        features[ind_0], features[ind_1]
    ])

    sens_processed = np.concatenate([
        np.zeros(len(ind_0)), np.ones(len(ind_1))
    ])
    
    labels_processed = np.concatenate([
        np.array((df['toxicity'] > ACC_BAR)[ind_0]), np.array((df['toxicity'] > ACC_BAR)[ind_1])
    ])

    n = features_processed.shape[0]
    random_ids = np.random.permutation(n)
    train_ids = random_ids[:int(0.8*n)]
    test_ids = random_ids[int(0.8*n):]
    
    train_features, train_labels, train_sensitives = features_processed[train_ids], labels_processed[train_ids], sens_processed[train_ids]
    test_features, test_labels, test_sensitives = features_processed[test_ids], labels_processed[test_ids], sens_processed[test_ids]
    
    train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.expand_dims(test_labels, 1)

    return train_features, test_features, train_labels, test_labels, train_sensitives, test_sensitives