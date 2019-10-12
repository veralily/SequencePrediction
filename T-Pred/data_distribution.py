import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

def plot_realvalue(pred, target):
    x = range(len(pred))
    total_width, n = 0.8, 2
    width = total_width / n

    plt.bar(x, [float(p) for p in pred], width=width, label='pred', fc='b')
    x = [e + width for e in x]
    plt.bar(x, [float(t) for t in target], width=width, label='target', fc='r')
    plt.legend()
    plt.show()
    return 0


def plot_distribution(pred, target):
    fig, ax = plt.subplots()

    plt.xlabel('Value of b')
    plt.ylabel('Probability')
    # plt.title(r'Distribution of Modulator b for Predicting Next Attribute on BPIC12')
    n, bins, patches1 = ax.hist(pred, bins=50, density=True, label='pred', facecolor='lightseagreen',
                                edgecolor="black")
    sns.kdeplot(pred, shade=True, color='lightseagreen')
    n, bins, patches2 = ax.hist(target, bins=50, density=True, label='target', facecolor='lightcoral',
                                edgecolor="black",
                                alpha=0.75)
    sns.kdeplot(target, shade=True, color='lightcoral')
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)

    legend = ax.legend(loc='upper center', fontsize='x-large')

    # plt.legend(handles = [ax1, ax2], labels = ['b_e', 'b_a'], fontsize = 'xx-large', loc = 'upper right')
    plt.show()


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


def to_percent(y, position):
    number = str(y).split('.')[0]
    if len(number) == 1:
        return '0.0' + str(y).split('.')[0]
    else:
        return '0.' + str(y).split('.')[0]


# style set
sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5)})

filename = 'output-RECSYS15-0821.txt'
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
            plot_realvalue(pred_t, target_t)
            plot_distribution(pred_t, target_t)
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