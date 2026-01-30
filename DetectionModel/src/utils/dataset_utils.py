import pandas as pd
import numpy as np
from pathlib import Path
from constants.detection.dataset_constants import RegModelConstants

def load_reg_dataset(reg_dataset_path=RegModelConstants.DATASET_METADATA_PATH):
    
    
    reg_df=pd.read_csv(reg_dataset_path)

    split_groups = set(reg_df[RegModelConstants.SPLIT_GROUP])
    
    features=reg_df.columns.to_list()
    bbox_features=list(filter(lambda feature:feature.startswith("bbox"),features))
    centorid_features=list(filter(lambda feature:feature.startswith("centroid"),features))
    volume_features=list(filter(lambda feature:feature.startswith("volume"),features))
    
    columns_to_drop = [
        RegModelConstants.FILE_NAME,
        RegModelConstants.PATIENT_ID,
        RegModelConstants.NOUDLE_INDEX,
        RegModelConstants.SLICE_INDEX,
        RegModelConstants.Features.FEATURE_ANNOTATION_COUNT,
        RegModelConstants.IMAGE_PATH,
        RegModelConstants.LABEL_PATH,
        *bbox_features,
        *centorid_features,
        *volume_features,
    ]
    
    reg_df.drop(columns=columns_to_drop,inplace=True)

    dataset={}

    for split in split_groups:

        split_df= reg_df[reg_df[RegModelConstants.SPLIT_GROUP]==split]

        x_features = split_df.drop(columns=[RegModelConstants.SPLIT_GROUP,
                                            RegModelConstants.Features.FEATURE_MALIGNANCY])    
        y_features = split_df[RegModelConstants.Features.FEATURE_MALIGNANCY]

        dataset[split]= {
            RegModelConstants.DATASET_X_FEATURES: x_features,
            RegModelConstants.DATASET_Y_FEATURES: y_features
        }
    
    return dataset





