import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _quantization_binning(data, num_bins=10):
        qtls = np.arange(0.0, 1.0 + 1 / num_bins, 1 / num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  
        return bin_edges, bin_centers, bin_widths

def _quantize(inputs, bin_edges, num_bins=10):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, num_bins) - 1  
        return quant_inputs

def _one_hot(a, num_bins=10):
    return np.squeeze(np.eye(num_bins)[a.reshape(-1).astype(np.int32)])

def DataQuantize(X, bin_edges=None, num_bins=10):    
    X_ = []
    for i in range(5):
        if bin_edges is not None:
            Xi_q = _quantize(X[:, i], bin_edges, num_bins)
        else:
            bin_edges, bin_centers, bin_widths = _quantization_binning(X[:, i], num_bins)
            Xi_q = _quantize(X[:, i], bin_edges, num_bins)
        Xi_q = _one_hot(Xi_q, num_bins)
        X_.append(Xi_q)

    for i in range(5, len(X[0])):
        if i == 39:     
            continue
        Xi_q = _one_hot(X[:, i], num_bins=2)
        X_.append(Xi_q)

    return np.concatenate(X_,1), bin_edges


def get_adult_data():
    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']

    train = pd.read_csv('/data/share/adult/raw/adult.data', header = None)
    test = pd.read_csv('/data/share/adult/raw/adult.test', header = None)
    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df['y'] = df['y'].replace({' <=50K.': 0, ' >50K.': 1, ' >50K': 1, ' <=50K': 0 })

    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
    df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

    delete_these = ['race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Black','race_ Other', 'sex_ Female']

    delete_these += ['native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong', 'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia']

    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)

    return BinaryLabelDataset(df = df, label_names = ['y'], protected_attribute_names = ['sex_ Male', 'race_ White'])



def preprocess_adult_data(seed=0, subsample_ratio=1.0, balancing_train_data=False):
    dataset_orig = get_adult_data()
    
    continuous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    continuous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continuous_features]
    
    if subsample_ratio < 1.0:
        n = dataset_orig.features.shape[0]
        subsample_index = np.random.choice(n, int(n * subsample_ratio), replace=False)
        dataset = dataset_orig.subset(subsample_index)
    else:
        dataset = dataset_orig
        
    dataset_train, dataset_test = dataset.split([0.8], shuffle=True, seed=seed)
    
    if balancing_train_data:
        
        gender_idx = 39
        s_train = dataset_train.features[:, gender_idx]  
        y_train = dataset_train.labels.flatten()  
        
        groups = {}
        for i, (y, s) in enumerate(zip(y_train, s_train)):
            key = (int(y), int(s))
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
            
        min_count = min(len(indices) for indices in groups.values())
        
        selected_indices = []
        for indices in groups.values():
            selected_indices.extend(np.random.choice(indices, min_count, replace=False))
            
        dataset_train = dataset_train.subset(selected_indices)
        np.random.shuffle(selected_indices)  

    SS = StandardScaler().fit(dataset_train.features[:, continuous_features_indices])
    dataset_train.features[:, continuous_features_indices] = SS.transform(dataset_train.features[:, continuous_features_indices])
    dataset_test.features[:, continuous_features_indices] = SS.transform(dataset_test.features[:, continuous_features_indices])

    X_train = dataset_train.features
    X_test = dataset_test.features

    y_train = dataset_train.labels
    y_test = dataset_test.labels
    
    A_train = X_train[:,39]
    A_test = X_test[:,39]

    X_train, bin_edges = DataQuantize(X_train)
    X_test, _ = DataQuantize(X_test, bin_edges)

    return X_train, X_test, y_train, y_test, A_train, A_test