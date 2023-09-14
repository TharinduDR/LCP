import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, f1_score, recall_score, precision_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


def rmse(preds, labels):
    return np.sqrt(((np.asarray(preds, dtype=np.float32) - np.asarray(labels, dtype=np.float32)) ** 2).mean())


def print_stat(data_frame, real_column, prediction_column):
    data_frame = data_frame.sort_values(real_column)

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (
        rmse_value, mae, pearson, spearman)

    print(textstr)


def print_binary_stat(df, real_column, pred_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    print()
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighter F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))
