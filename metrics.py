import tensorflow as tf
from sklearn import metrics as metrics_

# 其中 mask 是一个索引向量，值为1表示该位置的标签在训练数据中是给定的；比如100个数据中训练集已知带标签的数据有50个，
# 那么计算损失的时候，loss 乘以的 mask  等于 loss 在未带标签的地方都乘以0没有了，而在带标签的地方损失变成了mask倍；
# 即只对带标签的样本计算损失。
# 注：loss的shape与mask的shape相同，等于样本的数量：(None,），所以 loss *= mask 是向量点乘。
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))

    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def classify_evalue(y_true,y_pred,flag='acc'):
    """ classification evaluation index"""
    if flag=='acc':
        return metrics_.accuracy_score(y_true,y_pred)  #精度
    elif flag=='auc':
        fpr, tpr, thresholds = metrics_.roc_curve(y_true,y_pred)#计算ROC曲线的横纵坐标值，TPR，FPR TPR = TP/(TP+FN) = recall(真正例率，敏感度) FPR = FP/(FP+TN)(假正例率，1-特异性)
        return metrics_.auc(fpr, tpr) #ROC曲线下的面积;较大的AUC代表了较好的performance。
    elif flag == 'avg_auc':
        return metrics_.average_precision_score(y_true, y_pred, sample_weight=None)#根据预测得分计算平均精度(AP)
    elif flag == 'brier':
        return metrics_.brier_score_loss(y_true, y_pred, sample_weight=None, pos_label=None)#The smaller the Brier score, the better.
    elif flag == 'confusion':
        return metrics_.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)#通过计算混淆矩阵来评估分类的准确性 返回混淆矩阵
    elif flag == 'f1':
        return metrics_.f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)#F1值 F1 = 2 * (precision * recall) / (precision + recall) precision(查准率)=TP/(TP+FP) recall(查全率)=TP/(TP+FN)
    elif flag == 'log':
        return metrics_.log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)#对数损耗，又称逻辑损耗或交叉熵损耗
    elif flag == 'precision':
        return metrics_.precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary',) #查准率或者精度； precision(查准率)=TP/(TP+FP)
    elif flag == 'recall':
        return metrics_.recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)#查全率 ；recall(查全率)=TP/(TP+FN)
    elif flag == 'roc_auc':
        return metrics_.roc_auc_score(y_true, y_pred, average='macro', sample_weight=None)#计算ROC曲线下的面积就是AUC的值，the larger the better

