from data.datasets.adult     import preprocess_adult_data
from data.datasets.crime     import preprocess_crime_data
from data.datasets.dutch     import preprocess_dutch_data
from data.datasets.celeba_hq import preprocess_celeba_data
from data.datasets.civil     import preprocess_civil_roberta_data

DATA_PREPROCESSORS = {
    'adult' : preprocess_adult_data, 
    'crime' : preprocess_crime_data, 
    'dutch' : preprocess_dutch_data, 
    'celeba': preprocess_celeba_data,
    'civil_roberta' : preprocess_civil_roberta_data
}

def get_preprocessor(name: str):
    try:
        return DATA_PREPROCESSORS[name]
    except KeyError:
        raise ValueError(f"Unsupported dataset name: {name}")