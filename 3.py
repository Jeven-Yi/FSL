from collections import defaultdict
import numpy as np
# y_pred = [[0, 3, 4, 7, 'P', 0], [8, 9, 'L', 1]]
# y_true = [[0, 3, 4, 8, 'P', 0], [8, 9, 'L', 1]]
y_pred = [[0,3,'P',0],[4,8,'P',0],[8,9,'L',1]]
# y_pred = []
y_true = [[0,3,'P',0],[4,7,'P',0],[8,9,'M',1]]
labels = ['P','L','M']
def get_entities(seqs):
    res = [[] for _ in range(len(seqs))]
    for seq in seqs:
        res[seq[-1]].append(seq)
    vaild_cnt = 0
    # print(res)
    for i in range(len(res)):
        if len(res[i]) > 1:
            vaild_cnt += 1
            sort_tokens = sorted(res[i], key=lambda x: x[0])
            tmp = []
            for token in sort_tokens:
                tmp.extend(token[0:2])
            tmp.extend(sort_tokens[0][-2:])
            res[i] = tmp
        elif len(res[i]) == 1:
            vaild_cnt += 1
            res[i] = res[i][0]
        else:
            break
    res = res[:vaild_cnt]
    return res
print(get_entities(y_pred))
def extract_tp_actual_correct(y_true, y_pred, *args):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    if 1:#len(y_pred[0]) > 0 and isinstance(y_pred[0][-2], str):
        for token_true in get_entities(y_true):
            type_name = token_true[-2]
            position = token_true[:-2]
            entities_true[type_name].add(tuple(position))
        for token_pred in get_entities(y_pred):
            type_name = token_pred[-2]
            position = token_pred[:-2]
            entities_pred[type_name].add(tuple(position))
    else:
        pass
    if labels is not None:
        entities_true = {k: v for k, v in entities_true.items() if k in labels}
        entities_pred = {k: v for k, v in entities_pred.items() if k in labels}
    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    print(target_names)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum

pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true,y_pred)
print(pred_sum,tp_sum,true_sum)
# [0 1 1] [0 0 1] [1 0 1]