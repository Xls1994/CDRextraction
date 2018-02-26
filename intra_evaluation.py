# -*- coding: utf-8 -*-
'''
author:yangyl

'''
# -*- coding: utf-8 -*-
'''
author:yangyl

'''
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
def eval_p_r_f(file_path,label_path):

    y_true = []
    y_pred = []
    with open(label_path, 'r')as f:
        for line in f:
            line = line.strip()
            y_true.append(int(line))
    with open(file_path, 'r')as f:
        for line in f:
            line = line.strip().split(' ')
            if float(line[0]) > float(line[1]):
                y_pred.append(0)
            else:
                y_pred.append(1)
    return y_true,y_pred

def eval_dev_p_r_f(file_path,label_path):

    y_true = []
    y_pred = []
    with open(label_path, 'r')as f:
        for line in f:
            line = line.strip().split(' ')
            if float(line[0]) > float(line[1]):
                y_true.append(0)
            else:
                y_true.append(1)

    with open(file_path, 'r')as f:
        for line in f:
            line = line.strip().split(' ')
            if float(line[0]) > float(line[1]):
                y_pred.append(0)
            else:
                y_pred.append(1)
    return y_true,y_pred

def predict_test(predict_path,result_path,gold_path='./cdrEval/test.gold'):
    mapPrediction ={}
    p =open(predict_path,'r')
    gold =open(gold_path,'r')

    while True:

        predict_line =p.readline()
        predict_line =predict_line.strip()
        gold_line =gold.readline()
        gold_line =gold_line.strip()
        if  predict_line =='':
            break

        index =predict_line.split(' ')
        if float(index[1])>float(index[0]):
            val =float(index[1])
            mapPrediction[gold_line] = val

    with open(result_path,'w') as f:
        for key,val in mapPrediction.items():

            if key.find('-1')==-1 :
                f.write(key+'\t'+str(val)+'\n')
                # print key

def evaluate_for_corpus(f_path):
    import subprocess
    f1 = []
    dev_f1 = []

    files = f_path+'/'

    # fixed label path
    LABELPATH = './cdrEval/intra_label.nn'
    DEVPATH = files + 'dev_label.txt'
    evaluation = open(files + 'F-score.txt', 'w')
    evaluation.write("TestData\tDevData\n")
    for i in range(1, 11):
        file_path = files + 'result_' + str(i) + '.txt'
        # dev_file_path = files + 'dev_result_' + str(i) + '.txt'
        new_file_name = files + 'prediction_' + str(i) + '.txt'
        y_true, y_pred = eval_p_r_f(file_path, LABELPATH)

        # dev_y_true, dev_y_pred = eval_dev_p_r_f(dev_file_path, DEVPATH)

        ff = f1_score(y_true, y_pred, average='binary')
        # dev_ff = f1_score(dev_y_true, dev_y_pred, average='binary')
        f1.append(ff)
        # dev_f1.append(dev_ff)
        evaluation.write("*****" + str(i) + "****\n")
        # evaluation.write(str(ff) + "\t" + str(dev_ff) + '\n')

        predict_test(file_path, new_file_name)
        method = 'java -cp ./cdrEval/bc5cdr_eval.jar ncbi.bc5cdr_eval.Evaluate relation CID PubTator ' \
                 './cdrEval/CDR_TestSet.PubTator.txt ' + new_file_name + '>>' + files + 'predicion_result.txt'
        print 'evaluation for {} times'.format(i)
        subprocess.call(method, shell=True)
    evaluation.close()

    plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    plt.sca(ax1)
    plt.plot(range(1, 11), f1, '-og', label="F-score")

    # plt.legend(loc='lower right')
    # plt.sca(ax2)
    # plt.plot(range(1, 21), dev_f1, '-oy', label='Dev-F-score')
    # plt.legend(loc='lower right')
    # plt.savefig(files + '/MyFscore.jpg')


if __name__ == '__main__':
    print  'hello world'
    f_path ='./results/attVis'
    evaluate_for_corpus(f_path)

