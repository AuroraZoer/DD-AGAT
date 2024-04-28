import os
import re
import torch
import numpy as np
from sklearn.metrics import f1_score,classification_report

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def semeval_official_eval(label_map, preds, labels, outdir):
    proposed_answer = os.path.join(outdir, "proposed_answer.txt")
    answer_key  = os.path.join(outdir, "answer_key.txt")
    with open(proposed_answer, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(labels):
            f.write("{}\t{}\n".format(idx, label_map[pred]))
    with open(answer_key, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(idx, label_map[pred]))

    eval_cmd = "perl ./eval/semeval2010_task8_scorer-v1.2.pl {} {}".format(proposed_answer, answer_key)
    # print(eval_cmd)
    p,r,f1 = 0,0,0
    try:
        msg = [s for s in os.popen(eval_cmd).read().split("\n") if len(s) > 0]
        b_official = False
        for i,s in enumerate(msg):
            if "(9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL" in s:
                b_official = True
            if b_official is False:
                continue
            if "MACRO-averaged result (excluding Other)" in s and "F1 =" in msg[i+1]:
                p = float(re.findall('P = (.+?)%', msg[i+1])[0])
                r = float(re.findall('R = (.+?)%', msg[i+1])[0])
                f1 = float(re.findall('F1 = (.+?)%', msg[i+1])[0])
                break

    except Exception as e:
        print(str(e))
        f1 = 0
    print("p: {}, r: {}, f1: {}".format(p, r, f1))
    return {
        "precision": p,
        "recall": r,
        "f1": f1
    }

def write_prediction(relation_labels, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))

def compute_micro_f1(preds, labels, label_map, ignore_label, output_dir=None):
    if output_dir is not None:
        proposed_answer = os.path.join(output_dir, "proposed_answer.txt")
        answer_key = os.path.join(output_dir, "answer_key.txt")
        with open(proposed_answer, 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(labels):
                f.write("{}\t{}\n".format(idx, pred))
        with open(answer_key, 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(preds):
                f.write("{}\t{}\n".format(idx, pred))

    target_name = []
    target_id = []
    for name,id in label_map.items():
        if name in ignore_label:
            continue
        target_id.append(id)
        target_name.append(name)
    res = classification_report(labels, preds, labels=target_id, target_names=target_name, output_dict=True)
    print(res)
    if 'micro avg' in res:
        return res['micro avg']['f1-score']
    else:
        # 如果 'micro avg' 键不存在，使用 measure_statistics 函数进行计算
        rel_size = len(label_map) - (1 if ignore_label in label_map else 0)  # 根据 label_map 计算关系类别数量
        stats = measure_statistics(np.array(preds), np.array(labels), rel_size, label_map.get(ignore_label, -1))
        return stats['f1']  # 返回计算得到的微平均 F1 分数


def compute_metrics(preds, labels, rel_size, ignore_label):
    assert len(preds) == len(labels)
    # return acc_and_f1(preds, labels)
    return measure_statistics(preds, labels, rel_size, ignore_label)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='micro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
    }

def fbeta_score(precision, recall, beta=1.0):  # beta = 1 这对应于计算 F1 分数，即精确率和召回率同等重要
    # 计算 beta 的平方，存储在变量 beta_square 中
    beta_square = beta * beta
    # 检查精确率和召回率是否都不为零
    if (precision != 0.0) and (recall != 0.0):
        res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall))
    # 如果有一个或两个都为零，则将 F-beta 分数设为零
    else:
        res = 0.0
    return res

def measure_statistics(preds, labels, rel_size, ignore_label):
    """
    Calculate: True Positives (TP), False Positives (FP), False Negatives (FN)
    GPU & CPU code
    """

    # 将预测结果 preds 和真实标签 labels 转换为 PyTorch 张量
    y = torch.from_numpy(preds)
    t = torch.from_numpy(labels)

    # label_num 是一个大小为 1 的长整型张量，存储关系类别数量 rel_size
    label_num = torch.as_tensor([rel_size]).long()  
    # ignore_label 是一个大小为 1 的长整型张量，存储需要忽略的标签 ignore_label
    ignore_label = torch.as_tensor([ignore_label]).long()  

    # mask_t 是一个布尔型张量，用于判断真实标签是否与忽略标签相等
    mask_t = torch.eq(t, ignore_label)        # true = no_relation
    # mask_p 是一个布尔型张量，用于判断预测结果是否与忽略标签相等
    mask_p = torch.eq(y, ignore_label)        # pred = no_relation

    # true 是一个张量，用于存储替换忽略标签后的真实标签
    true = torch.where(mask_t, label_num, t)  # t: ground truth labels (replace ignored with +1)
    # pred 是一个张量，用于存储替换忽略标签后的预测结果
    pred = torch.where(mask_p, label_num, y)  # y: output of neural network (replace ignored with +1)

    # tp_mask 表示预测结果与真实标签相等的情况
    tp_mask = torch.where(torch.eq(pred, true), true, label_num)
    # fp_mask 表示预测结果与真实标签不相等的情况，包括错误的正类
    fp_mask = torch.where(torch.ne(pred, true), pred, label_num)  # this includes wrong positive classes as well
    # fn_mask 表示预测结果与真实标签不相等的情况
    fn_mask = torch.where(torch.ne(pred, true), true, label_num)

    # 使用 torch.bincount 函数计算每个类别的 TP、FP 和 FN 数量
    tp = torch.bincount(tp_mask, minlength=rel_size + 1)[:rel_size]
    fp = torch.bincount(fp_mask, minlength=rel_size + 1)[:rel_size]
    fn = torch.bincount(fn_mask, minlength=rel_size + 1)[:rel_size]
    tn = torch.sum(mask_t & mask_p)

    # atp 是 TP 的总和
    atp = np.sum(tp.numpy())
    # afp 是 FP 的总和
    afp = np.sum(fp.numpy())
    # afn 是 FN 的总和
    afn = np.sum(fn.numpy())
    # atn 是 TN 的总和
    atn = np.sum(tn.numpy())
    # micro_p 是精确率，计算公式为 TP / (TP + FP)
    micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
    # micro_r 是召回率，计算公式为 TP / (TP + FN)
    micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
    # micro_f 是 F1 值，使用 fbeta_score 函数计算
    micro_f = fbeta_score(micro_p, micro_r)

    return {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f,
    }



