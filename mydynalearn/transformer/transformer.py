import torch

def TP_to_class(TP):
    return TP.max(-1)[1]

def dict_to_tensor(dict):
    for key in dict.keys():
        dict