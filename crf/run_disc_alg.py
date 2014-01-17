'''
Created on Oct 28, 2013

@author: samshuster
'''
from CRF import CRF
from disc_algs import DiscAlg
import features

TRAIN_DATA = 'train.tags'
DEV_DATA = 'dev.tags'
TEST_DATA  = 'test.tags'

def load_data(f,amount=float('inf')):
    x = []
    y = []
    with open(f,'r') as inp:
        for ind,line in enumerate(inp):
            if ind >= amount:
                break
            groups = line.split(' ')
            sentence_x = []
            sentence_y = []
            for group in groups:
                try:
                    word,pos = group.split('_')
                    sentence_x.append(word.strip())
                    sentence_y.append(pos.strip())
                except ValueError: pass
            x.append(sentence_x)
            y.append(sentence_y)
    return x,y

def train(c,f,x,y,xd,yd):
    c.fit(x,y,xd,yd)
    c.save(f)

def crf():
    for reg in [0.3]:
        f = 'crf_q5{}'.format(reg)
        facc = 'crf_q5{}'.format(reg)
        x,y = load_data(TRAIN_DATA)
        xd,yd = load_data(DEV_DATA)
        xte, yte = load_data(TEST_DATA)
        c = DiscAlg(out_file=f,acc_out_file=facc,reg = reg,alg='crf')
        train(c,f,x,y,xd,yd)
            
def perceptron(amount,reg,step,step_dec):
    f = 'perceptron_q4_reg{}step{}dec{}'.format(reg,step,step_dec)
    facc = 'perceptron_q4_results_reg{}step{}dec{}'.format(reg,step,step_dec)
    x,y = load_data(TRAIN_DATA,amount)
    xd,yd = load_data(DEV_DATA)
    xte, yte = load_data(TEST_DATA)
    c = DiscAlg(out_file=f,acc_out_file=facc, reg = reg,step_dec = step_dec,  step_size = step, alg = 'perceptron')
    train(c,f,x,y,xd,yd)
        
def average_perceptron(amount,reg,step,step_dec):
    f = 'avgperceptron_q5_reg{}step{}dec{}'.format(reg,step,step_dec)
    facc = 'avgperceptron_q5_results_reg{}step{}dec{}'.format(reg,step,step_dec)
    x,y = load_data(TRAIN_DATA,amount)
    xd,yd = load_data(DEV_DATA)
    xte, yte = load_data(TEST_DATA)
    c = DiscAlg(out_file=f,acc_out_file=facc,step_dec = step_dec,  step_size = step, reg = reg,alg = 'avgperceptron')
    train(c,f,x,y,xd,yd)

def output_pred(out_file, x, y):
    with open(out_file, 'w') as out:
        for x_i, y_i in zip(x,y):
            if x_i == 'START' or x_i == 'END':
                continue
            for x_ii, y_ii in zip(x_i,y_i):
                out.write('{0}_{1} '.format(x_ii,y_ii))
        out.write('\n')
        
if __name__ == '__main__':
    #crf()
    amount = float(50)
    step = 1
    step_dec = 0
    reg = 0
    for step in [0.01, 0.1,1,10,100]:
        perceptron(amount,reg,step,step_dec)
    '''
    for reg in [100]:
        perceptron(amount,reg,step,step_dec)
    ret = 0.01
    for step in [0.01, 0.1, 0.5, 1]:
        perceptron(amount, reg, step,step_dec)
        average_perceptron(amount,reg,step,step_dec)
    reg = 0.01
    for step_dec in [0, 1.5,0.75,0.25,0.01]:
        perceptron(amount,reg,step,step_dec)
        average_perceptron(amount,reg,step,step_dec)

    f = 'perceptron_q4_reg0.001'
    facc = 'perceptron_q4_reg0.001_results'
    step_dec = 0.25
    step_size = 4
    c = DiscAlgs(out_file=f,acc_out_file=facc, reg = reg,step_dec = step_dec,  step_size = step, alg = 'perceptron')
    x,y = load_data(TRAIN_DATA,amount)
    xd,yd = load_data(DEV_DATA)
    xte, yte = load_data(TEST_DATA)
    c.load(f,x,y)
    y_pred = c.predict(xd,True)
    output_pred('dev_pred.tags',xd,y_pred)
    print c.calc_accuracy(y_pred,yd)
    yt_pred = c.predict(xte,True)
    print c.calc_accuracy(yt_pred,yte)
    output_pred('test_pred.tags',xte,yt_pred)
    '''

