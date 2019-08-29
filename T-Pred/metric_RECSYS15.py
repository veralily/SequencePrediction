import os
import numpy as np
import math
import matplotlib.pyplot as plt

def plot(pred, target):
    x = range(len(pred))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, [float(p) for p in pred], width=width, label='pred', fc='b')
    x = [e + width for e in x]
    plt.bar(x, [float(t) for t in target], width=width, label='target', fc='r')
    plt.legend()
    plt.show()
    return 0


def accuracy(pred, target):
    count = 0.0
    if len(pred) == len(target):
        sum = len(pred)
    else:
        sum = 0
        print('length of pred and target for computing accu is not equal!!')
    for p, t in zip(pred, target):
        if p == t:
            count += 1
        else:
            pass
    return count/float(sum)


def mae(pred, target):
    sum = 0.0
    if len(pred) == len(target):
        number = len(pred)
    else:
        number = 0
        print('length of pred and target for computing mae is not equal!!')
        print('length of pred: %d' % (len(pred)))
        print('length of target: %d' % (len(target)))
    for p, t in zip(pred, target):
        ae = math.fabs(float(p) - float(t))
        if float(t) == 0:
            t = 0.1
        # sum = sum + ae / float(t)
        sum = sum + ae
    return sum / float(number)


filename = 'output-cikm16-0826.txt'
# filename = 'output-CIKM16-0716.txt'


with open(filename) as f:
    lines = f.readlines()
    pred_t = []
    mean_abs_error = []
    for line in lines:
        if 'pred_e' in line:
            content = line.replace('\n', '').split(': ')[1].split('\t')
            pred_e = content
        elif 'targ_e' in line:
            content = line.replace('\n', '').split(': ')[1].split('\t')
            target_e = content
        elif 'pred_t' in line:
            content = line.replace('\n', '').split(': ')[1].replace('[', '').replace(']', '').replace(' ', '')
            pred_t.append(content)
        elif 'targ_t' in line:
            content = line.replace('\n', '').split(': ')[1].split('\t')
            target_t = content
            print('Accuracy: %f' % (accuracy(pred_e, target_e)))
            plot(pred_t, target_t)
            # print(pred_t)
            mean_abs_error.append(mae(pred_t, target_t))
            print('MAE: %f' % (mae(pred_t, target_t)))
            pred_t = []


        else:
            content = line.split('\t')
            if len(content) == 1:
                content = content[0].split(']')[0].replace('[', '').replace(' ', '')
                pred_t.append(content)
            else:
                for c in content:
                    c = c.split(']')[0].replace('[', '').replace(' ', '')
                    pred_t.append(c)

    print('Final MAE for all test data: %f' % (math.fsum(mean_abs_error)/float(len(mean_abs_error))))

