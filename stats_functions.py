
def precision(TP, FP):
    return TP/(TP+FP)

def recall(TP, FN):
    return TP/(TP+FN)

def F1_score(prec, rec):
    return prec * rec / (prec + rec)
    