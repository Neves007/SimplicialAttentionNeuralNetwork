import torch

def TP_to_class(TP):
    return TP.max(-1)[1]



