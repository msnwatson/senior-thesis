import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def viz_time_series(data, labels=None):
    x = []
    y = []
    c = []
    i = 0
    for _, v in data.items():
        for t in v:
            x.append(t)
            y.append(i)
            if labels != None:
                c.append(labels[i])
        i += 1

    if labels != None:
        plt.scatter(x, y, c=c)
    else:
        plt.scatter(x, y)
    plt.xlabel("Normalized Time")
    plt.ylabel("Account Number")
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.show()

def average_classification_uncertainty(em_clusterer):
    pred_labels = em_clusterer.classify()
    avg_uncertainty = defaultdict(lambda : [])
    
    for i in range(len(pred_labels)):
        avg_uncertainty[pred_labels[i]].append(1 - np.max(em_clusterer.z[i, :]))
         
    
    for k, v in avg_uncertainty.items():
        avg_uncertainty[k] = sum(v) / len(v)
    
    return avg_uncertainty


def bot_proportion_per_cluster(ids, pred_labels, bot_labels_df):
    bot_count_by_label = defaultdict(lambda : 0)
    total_count_by_label = defaultdict(lambda : 0)
    
    for i in range(len(pred_labels)):
        total_count_by_label[pred_labels[i]] += 1
        
        if (bot_labels_df[bot_labels_df['id'] == ids[i]]['label'] == 'bot').all():
            bot_count_by_label[pred_labels[i]] += 1
        
    ret_lst = []
    for k, v in bot_count_by_label.items():
        ret_lst.append(v / total_count_by_label[k])
        
    return ret_lst, total_count_by_label

def first_event_time(data):
    earliest_per_acc = []
    for _, v in data.items():
        earliest_per_acc.append(min(v))

    return min(earliest_per_acc)

def normalize_data(data, T):
    time_zero = first_event_time(data)
    for k, v in data.items():
        norm_lst = []

        for t in v:
            norm_lst.append(t - time_zero)

        norm_lst.sort()
        data[k] = norm_lst

    return data, T - time_zero