from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from metrics import classify_evalue
import numpy as np
import random
def get_list():
    dataset='R8'
    f = open('../data/corpus/' + dataset + '.clean.txt', 'r')
    data = f.readlines()

    f = open('../data/' + dataset + '_shuffle.txt', 'r')
    lines = f.readlines()
    datas=[]
    labels=[]
    test_datas=[]
    test_labels=[]
    for line in lines:
        t=line.split('\t')
        if t[1]=='train':
            datas.append(data[int(t[0])])
            labels.append(int(t[2][0]))
        if t[1]=='test':
            test_datas.append(data[int(t[0])])
            test_labels.append(int(t[2][0]))
    f.close()
    data_lst=[]
    for data in datas:
        t=data.split()
        data_lst.append(' '.join(t))

    test_data_lst=[]
    for data in test_datas:
        t=data.split()
        test_data_lst.append(' '.join(t))

    return data_lst,labels,test_data_lst,test_labels

def proceed_dict1():
    data, target,test_data_lst,test_labels=get_list()
    train_num=7108
    train_withdraw=int(train_num*0.025)
    idx = random.sample(range(0, train_num), train_withdraw)

    X_train=[data[i] for i in idx]
    y_train=[target[i] for i in idx]
    X_test=test_data_lst
    y_test=test_labels

    count_vec = CountVectorizer()
    transformer = TfidfTransformer()

    # 只使用词频统计的方式将原始训练和测试文本转化为特征向量。
    # X_count_train = count_vec.fit_transform(X_train)
    X_count_train = transformer.fit_transform(count_vec.fit_transform(X_train))
    X_count_test = transformer.transform(count_vec.transform(X_test))
    clf = linear_model.LogisticRegression()
    # clf = MultinomialNB()
    # clf = SVC(kernel='linear')  # SVM模块，svc,线性核函数
    clf.fit(X_count_train, y_train)
    # print(clf.score(X_count_test,y_test))
    y_pred=clf.predict(X_count_test)
    print(classify_evalue(y_test,y_pred, 'acc'))
    # print(classify_evalue(y_test,y_pred, 'f1'))
    # print(classify_evalue(y_test,y_pred, 'roc_auc'))

for _ in range(10):
    proceed_dict1()

