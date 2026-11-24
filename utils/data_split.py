import numpy as np

data_info = {
# EASY
    "civit_easy_01": {
        "type": "train",
        "date": "march"
    },
    "civit_easy_02": {
        "type": "train",
        "date": "april2"
    },
    "civit_easy_03": {
        "type": "test",
        "date": "april2"
    },
    "civit_easy_04": {
        "type": "train",
        "date": "april2"
    },
    "civit_easy_05": {
        "type": "train",
        "date": "april2"
    },
    "civit_easy_06": {
        "type": "train",
        "date": "april2"
    },
    "civit_easy_prashant_02": {
        "type": "test",
        "date": "april3"
    },
    "civit_easy_prashant_03": {
        "type": "test",
        "date": "april4"
    },
    "civit_easy_nikolay_01": {
        "type": "test",
        "date": "april4"
    },

# NORM
    "civit_norm_01": {
        "type": "train",
        "date": "march"
    },
    "civit_norm_02": {
        "type": "train",
        "date": "march"
    },
    "civit_norm_03": {
        "type": "train",
        "date": "march"
    },
    "civit_norm_04": {
        "type": "train",
        "date": "march"
    },
    "civit_norm_05": {
        "type": "test",
        "date": "march"
    },
    "civit_norm_06": {
        "type": "train",
        "date": "april"
    },
    "civit_norm_08": {
        "type": "train",
        "date": "april"
    },
    "civit_norm_10": {
        "type": "train",
        "date": "april2"
    },
    "civit_norm_11": {
        "type": "test",
        "date": "april2"
    },
    "civit_norm_12": {
        "type": "train",
        "date": "april3"
    },
    "civit_norm_prashant_01": {
        "type": "test",
        "date": "april3"
    },
    "civit_norm_prashant_02": {
        "type": "test",
        "date": "april3"
    },
    # "civit_norm_prashant_03": {
    #     "type": "test",
    #     "date": "april4"
    # },
    "civit_norm_nikolay_01": {
        "type": "test",
        "date": "april4"
    },


# HARD
    "civit_hard_01": {
        "type": "train",
        "date": "april"
    },
    "civit_hard_02": {
        "type": "train",
        "date": "april"
    },
    "civit_hard_03": {
        "type": "test",
        "date": "april3"
    },
    "civit_hard_04": {
        "type": "train",
        "date": "april3"
    },
    "civit_hard_05": {
        "type": "train",
        "date": "april3"
    },

# MULTI
    "civit_multi_01": {
        "type": "train",
        "date": "april"
    },
    "civit_multi_02": {
        "type": "test",
        "date": "april"
    },
    # "civit_multi_03": {
    #     "type": "train",
    #     "date": "april4"
    # },
    "civit_multi_04": {
        "type": "train",
        "date": "april4"
    },
    "civit_multi_05": {
        "type": "train",
        "date": "april4"
    },
    "civit_multi_06": {
        "type": "train",
        "date": "april4"
    },
    "civit_multi_07": {
        "type": "test",
        "date": "april4"
    },
}