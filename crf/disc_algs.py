'''
Created on Oct 29, 2013

@author: samshuster
'''

from random import shuffle
from random import seed
import numpy as np
import math
import random
import features
import cPickle as pickle
from collections import defaultdict
import time
import matplotlib.pyplot as plt

class DiscAlg(object):
    '''
    classdocs
    '''

    START = 'START'
    END = 'END'
    TAG = '!T'
    PERCEPTRON = 'perceptron'
    AVERAGE_PERCEPTRON = 'avgperceptron'
    CRF = 'crf'

    def __init__(self, func_list = [], step_size = 4,step_dec = 0.25,
                  reg = 0.1,max_iter=4, acc_threshold = 0.90,out_file = None,
                  acc_out_file = None,alg='perceptron'):
        '''
        Create a CRF from a dictionary
        '''
        seed(2)
        self.func_list = func_list
        self.step_size = step_size
        self.reg = reg
        self.max_iter = max_iter
        self.step_dec = step_dec
        self.acc_threshold = acc_threshold
        self.out_file = out_file
        self.out_acc_file = acc_out_file
        self.unique_y = ['START', 'PRP$', 'VBG', 'VBD', 'VBN', ',', "''", 'VBP', 'WDT', 'JJ', 'WP', 
         'VBZ', 'DT', '#', 'RP', '$', 'NN', 'FW', 'POS', '.', 'TO', 'PRP', 'RB', '-LRB-', ':',
          'NNS', 'NNP', '``', 'WRB', 'CC', 'LS', 'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN', 'WP$',
           'MD', 'NNPS', '-RRB-', 'JJS', 'JJR', 'SYM', 'VB', 'UH', 'END']
        self.alg = alg

  
    def calc_accuracy(self, pred, actual):
        for yp, ya in zip(pred,actual):
            print yp
            print ya
        accuracy = [(reduce(lambda x,(yp,ya): x+1 if yp == ya else x,zip(pred_j[1:-1],actual_j),0))/float(len(actual_j))
                     for pred_j,actual_j in zip(pred,actual)]
        return sum(accuracy) / float(len(accuracy))
    
    def fit(self, x, y,xd,yd):
        '''
        @param x: A list of a sequence of inputs (ordered, list of list)
        @param y: A list of a sequence of outputs (ordered, list of list)
        @param func_list: A list of functions that take two inputs, a list of inputs, a list of outputs
        '''
        #Will add START and END to each of the y sequences, meaning that len(y_j) = len(x_j)+2
        self.preprocess_data(x,y)
        self.infer_features(x,y)
        self.lambdas = [0 for _ in self.func_list]
        print len(self.lambdas)
        #self.func_cache = self.create_func_cache(x)
        t_acc = 0
        if self.out_acc_file:
            with open(self.out_acc_file,'a') as out:
                out.write('---PARAM----\n')
                out.write('reg constant:\t{}\n'.format(self.reg))
                out.write('initial step:\t{}\n'.format(self.step_size))
                out.write('step_decrease:\t{}\n'.format(self.step_dec))
                out.write('acc_threshold:\t{}\n'.format(self.acc_threshold))
                out.write('max_iter:\t{}\n'.format(self.max_iter))
                out.write('num_feat:\t{}\n'.format(len(self.func_list)))
                out.write('alg:\t{}\n'.format(self.alg))
                out.write('---RESULTS----\n')
        if self.alg == self.PERCEPTRON:
            print 'USING PERCEPTRON'
            sgd_iter = self.sgd(x,y)
        elif self.alg == self.AVERAGE_PERCEPTRON:
            print 'USING AVG PERCEPTRON'
            sgd_iter = self.sgd_averaged(x, y)
        else:
            print 'USING CRF'
            sgd_iter = self.sgd_crf(x,y)
        while t_acc < self.acc_threshold:
            try:
                num,step,it,t_acc = sgd_iter.next()
            except StopIteration as e:
                print 'Did not converge'
                break
            yd_pred = self.predict(xd,True)
            d_acc = self.calc_accuracy(yd_pred,yd)
            if self.out_acc_file:
                with open(self.out_acc_file,'a') as out:
                    out.write('num trained:\t{}\n'.format(num))
                    out.write('cur stepsize:\t{}\n'.format(step))
                    out.write('cur iteration:\t{}\n'.format(it))
                    out.write('train accuracy:\t{}\n'.format(t_acc))
                    out.write('dev accuracy:\t{}\n'.format(d_acc))
            if self.out_file:
                self.save(self.out_file)
        plt.savefig(self.out_acc_file+'.png', bbox_inches='tight')
    
    def save(self,out_file):
        out = [self.lambdas,self.unique_y]
        pickle.dump(out,open(out_file,'wb'))
       
    def load(self,out_file, x, y):
        [self.lambdas,self.unique_y] = pickle.load(open(out_file,'rb'))
        self.preprocess_data(x,y)
        self.infer_features(x,y)
        
    def predict(self, x,insert=False):
        pred_y = []
        print len(self.unique_x)
        for x_j in x:
            if insert and x_j[0] != self.START:
                x_j.insert(0,self.START)
                x_j.append(self.END)
            if self.alg == self.PERCEPTRON or self.alg == self.AVERAGE_PERCEPTRON:
                matrices_j = self.create_transition_dicts(x_j)
                best_tag = self.viterbi_dict_search(x_j, matrices_j)
            else:
                matrices_j = self.create_transition_dicts_crf(x_j)
                best_tag = self.viterbi_dict_search_crf(x_j, matrices_j)
            pred_y.append(best_tag)
        return pred_y
    
    def preprocess_data(self, x, y):
        '''Generates features and returns a list of dictionaries'''
        self.unique_x = set()
        ty_unique = set()
        for x_j,y_j in zip(x,y):
            self.unique_x = self.unique_x.union(set(_ for _ in x_j))
            ty_unique = ty_unique.union(set(_ for _ in y_j))
            y_j.insert(0,self.START)  
            y_j.append(self.END)
            x_j.insert(0,self.START)
            x_j.append(self.END)
        self.unique_x = list(self.unique_x)
        ty_unique = list(ty_unique)
        ty_unique.insert(0,self.START)
        ty_unique.append(self.END)
        self.unique_x.insert(0,self.START)
        self.unique_x.append(self.END)
        if not self.unique_y:
            self.unique_y = ty_unique
        
    def infer_features(self, x, y):
        self.func_filter_dict = {}
        self.reg_exp = []
        reg_exp ='ct{}pt{}'
        self.reg_exp.append(lambda x_j, cur_tag, prev_tag, i,reg_exp=reg_exp: reg_exp.format(cur_tag,prev_tag))
        for y1 in self.unique_y:
            for y2 in self.unique_y:
                new_func = lambda words,curr_tag,prev_tag,i,y1=y1,y2=y2 : features.bigram_tag_label(words, curr_tag, prev_tag, i, tag=y1,tag2=y2)
                self.func_list.append(new_func)
                self.func_filter_dict[reg_exp.format(y1,y2)] = len(self.func_list) - 1

        for ind in range(-1,2):
            reg_exp = 'ct{}w{}i{}'
            self.reg_exp.append(lambda x_j, cur_tag, prev_tag, i,reg_exp=reg_exp,ind=ind: reg_exp.format(cur_tag,x_j[i-ind].lower(),ind))
            for y1 in self.unique_y:
                for x1 in self.unique_x:
                    new_func = lambda words,curr_tag,prev_tag,i,y1=y1,x1=x1,ind=ind : features.atomic_tag_label(words, curr_tag, prev_tag, i-ind, tag=y1,word=x1)
                    self.func_list.append(new_func)
                    self.func_filter_dict[reg_exp.format(y1,x1,ind)] = len(self.func_list) - 1

        for slength in range(3,5):
            reg_exp = 'ct{}s{}l{}'
            self.reg_exp.append(lambda x_j, cur_tag, prev_tag, i,l=slength,reg_exp=reg_exp: reg_exp.format(cur_tag,x_j[i][-l:].lower(),l))
            for y1 in self.unique_y:
                for x1 in self.unique_x:
                    s = x1[-slength:]
                    new_func = lambda words,curr_tag,prev_tag,i,suff_length=slength,suffix=s,tag=y1 : features.suffix(words, curr_tag, prev_tag, i, suff_length=suff_length,suffix=suffix,tag=y1)
                    self.func_list.append(new_func)
                    self.func_filter_dict[reg_exp.format(y1,s,slength)] = len(self.func_list) - 1

                
    def update_line(self, hl,ax,x_data,y_data):
        hl.set_xdata(np.append(hl.get_xdata(), x_data))
        hl.set_ydata(np.append(hl.get_ydata(), y_data))
        ax.relim()
        ax.autoscale_view()
        plt.draw()

    def sgd_averaged(self,x,y):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.set_window_title('Training Accuracy versus Update Number')
        hl, = ax.plot([], [],'-k',label='black')
        num = 0
        for it in range(self.max_iter):
            self.c = 0
            self.lambdas_prime = [0 for _ in self.func_list]
            print it
            accuracy = []
            order = range(len(y))
            shuffle(order)
            for exp_ind,j in enumerate(order):
                x_j = x[j]
                y_j = y[j]
                matrices_j = self.create_transition_dicts(x_j)
                best_tag = self.viterbi_dict_search(x_j, matrices_j)
                print y_j
                print best_tag
                funcs = set()
                for rexp in self.reg_exp:
                    for i in range(1,len(y_j)-1):
                        try:
                            func_ind = self.func_filter_dict[rexp(x_j,y_j[i],y_j[i-1],i)]
                            funcs.add(func_ind)
                        except KeyError as e: pass
                        try:
                            func_ind = self.func_filter_dict[rexp(x_j,best_tag[i],best_tag[i-1],i)]
                            funcs.add(func_ind)
                        except KeyError as e: pass
                n = 0
                for ind,(lamb,lamb_prime) in enumerate(zip(self.lambdas,self.lambdas_prime)):
                    self.lambdas[ind] = lamb - lamb*self.step_size*self.reg
                    self.lambdas_prime[ind] = lamb_prime - lamb_prime*self.c*self.step_size*self.reg
                    n = n + lamb**2
                print 'Weight Norm:', n
                for func_ind in funcs:
                    param_i = self.lambdas[func_ind]
                    func_i = self.func_list[func_ind]
                    optimal_val = 0
                    guessed_val = 0
                    for i in range(1,len(y_j)-1):
                        optimal_val += func_i(x_j,y_j[i],y_j[i-1],i)
                        guessed_val += func_i(x_j,best_tag[i],best_tag[i-1],i)
                    self.lambdas[func_ind] += self.step_size*(optimal_val - guessed_val)
                    self.lambdas_prime[func_ind] += self.c*self.step_size*(optimal_val - guessed_val)
                    self.c += 1
                acc = reduce(lambda x,(yp,ya): x+1 if yp == ya else x,
                             zip(best_tag[1:-1],y_j[1:-1]),0)/float(len(y_j[1:-1]))
                print acc
                accuracy.append(acc)
                self.update_line(hl,ax, num,acc)
                num += 1
            self.step_size *= self.step_dec
            t_acc = sum(accuracy)/float(len(accuracy))
            for ind, (lamb,lamb_prime) in enumerate(zip(self.lambdas,self.lambdas_prime)):
                self.lambdas[ind] = self.lambdas[ind] - self.lambdas_prime[ind] / float(self.c)
            yield num, self.step_size, it,t_acc

    def sgd(self,x,y):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.set_window_title('Training Accuracy versus Update Number')
        hl, = ax.plot([], [],'-k',label='black')
        num = 0
        for it in range(self.max_iter):
            print it
            accuracy = []
            order = range(len(y))
            shuffle(order)
            for exp_ind,j in enumerate(order):
                x_j = x[j]
                y_j = y[j]
                matrices_j = self.create_transition_dicts(x_j)
                best_tag = self.viterbi_dict_search(x_j, matrices_j)
                print y_j
                print best_tag
                funcs = set()
                for rexp in self.reg_exp:
                    for i in range(1,len(y_j)-1):
                        try:
                            func_ind = self.func_filter_dict[rexp(x_j,y_j[i],y_j[i-1],i)]
                            funcs.add(func_ind)
                        except KeyError as e: pass
                        try:
                            func_ind = self.func_filter_dict[rexp(x_j,best_tag[i],best_tag[i-1],i)]
                            funcs.add(func_ind)
                        except KeyError as e: pass
                n = 0
                for ind,lamb in enumerate(self.lambdas):
                    self.lambdas[ind] = lamb - lamb*self.step_size*self.reg
                    n = n + lamb**2
                print 'Weight Norm:', n
                for func_ind in funcs:
                    param_i = self.lambdas[func_ind]
                    func_i = self.func_list[func_ind]
                    optimal_val = 0
                    guessed_val = 0
                    for i in range(1,len(y_j)-1):
                        optimal_val += func_i(x_j,y_j[i],y_j[i-1],i)
                        guessed_val += func_i(x_j,best_tag[i],best_tag[i-1],i)
                    self.lambdas[func_ind] += self.step_size*(optimal_val - guessed_val)
                acc = reduce(lambda x,(yp,ya): x+1 if yp == ya else x,
                             zip(best_tag[1:-1],y_j[1:-1]),0)/float(len(y_j[1:-1]))
                print acc
                accuracy.append(acc)
                self.update_line(hl,ax, num,acc)
                num += 1
            self.step_size *= self.step_dec
            t_acc = sum(accuracy)/float(len(accuracy))
            yield num, self.step_size, it,t_acc
    
    def viterbi_dict_search(self,x_j,trans_dicts):
        #works on the less memory efficient dictionaries but clearer for debugging
        Q = []
        best_prev = []
        #Init Q
        for word_ind, word in enumerate(x_j):
            bp = {}
            QT = {}
            if word == self.START: pass
            elif word == self.END: pass
            elif word_ind == 1:
                for cur_tag in self.unique_y:
                    QT[cur_tag] = trans_dicts[word_ind][cur_tag][self.START]
            else:
                for curtag in trans_dicts[word_ind]:
                    best_score = 0
                    for prevtag_ind, prevtag in enumerate(trans_dicts[word_ind][curtag]):
                        r = trans_dicts[word_ind][curtag][prevtag] + Q[word_ind-1][prevtag]
                        if prevtag_ind == 0:
                            best_score = r
                            bp[curtag] = prevtag
                            QT[curtag] = r
                        elif r > best_score:
                            best_score = r
                            bp[curtag] = prevtag
                            QT[curtag] = r
            Q.append(QT)            
            best_prev.append(bp)
        best_prev.reverse()
        s = 0
        s_tag = ''
        QT = Q[-2]
        for curtag_ind, curtag in enumerate(QT):
            if curtag_ind == 0:
                s = QT[curtag]
                s_tag = curtag
            elif QT[curtag] > s:
                s = QT[curtag]
                s_tag = curtag
        ctag = s_tag
        tags = [s_tag]
        for BP in best_prev[1:-2]:
            ctag = BP[ctag]
            tags.insert(0,ctag)
        tags.insert(0,self.START)
        tags.append(self.END)
        return tags
            
            
    def viterbi_search(self,x_j,trans_matrices):
        #previous on the rows
        #current on the columns
        #x_j is augmented with START and END
        N_2 = len(x_j)
        N_1 = N_2 - 1
        N = N_1 - 1
        avocab_size = len(self.unique_y)
        vocab_size = avocab_size - 2
        #Create the memory table of size only the proper tags, and all possible sequence pairs
        Q = np.zeros([len(self.unique_y),N])
        best_prev = np.zeros([len(self.unique_y,N)])
        #Base case
        for i in range(1,vocab_size-1):
            Q[i-1,0] = trans_matrices[0][i]
        #Now starting at cur_word = 2, prev_word = 1
        for cur_x in range(1,N):
            for cur_ind in range(1,avocab_size-1):
                Q[cur_ind-1,cur_x] = 0
                best_prev[cur_ind-1,cur_x] = 0
                best_score = float('-inf')
                for prev_ind in range(1,avocab_size-1):
                    if cur_x == N_1:
                        r = trans_matrices[cur_x][prev_ind] + Q[prev_ind-1,cur_x-1]
                    else:
                        r = trans_matrices[cur_x][prev_ind,cur_ind]+Q[prev_ind-1,cur_x-1]
                    if r > best_score:
                        best_score = r
                        best_prev[cur_ind-1,cur_x] = prev_ind-1
                        Q[cur_ind-1,cur_x] = r
        final_best = 0
        final_score = float('-inf')
        for j in range(1,avocab_size-1):
            if Q[j-1,N-1] > final_score:
                final_score = Q[j-1,N-1]
                final_best = j-1
        best_tag = [self.unique_y[final_best+1]]
        current = final_best
        for i in range(N-2,-1,-1):
            current = int(best_prev[int(current),i+1])
            best_tag.insert(0,self.unique_y[current+1])
        best_tag.insert(0,self.START)
        best_tag.append(self.END)
        print best_tag
        print x_j
        return best_tag

        
    def create_transition_dicts(self,x_j):
        transition_dicts = []
        for word_ind,word in enumerate(x_j):
            if word == self.START:
                transition_dicts.append(self.START)
            elif word == self.END:
                transition_dicts.append(self.END)
            else:
                wdict = defaultdict(lambda: defaultdict(float))
                for curtag in self.unique_y:
                    for prevtag in self.unique_y:
                        weight = 0
                        for rexp in self.reg_exp:
                            try:
                                func_ind = self.func_filter_dict[rexp(x_j,curtag,prevtag,word_ind)]
                                param = self.lambdas[func_ind]
                                func = self.func_list[func_ind]
                                weight += param*func(x_j,curtag,prevtag,word_ind)
                            except KeyError as e: pass
                        wdict[curtag][prevtag] = weight
                transition_dicts.append(wdict)
        return transition_dicts

    def create_transition_dicts_crf(self,x_j):
        transition_dicts = []
        for word_ind,word in enumerate(x_j):
            if word == self.START:
                transition_dicts.append(self.START)
            elif word == self.END:
                transition_dicts.append(self.END)
            else:
                wdict = defaultdict(lambda: defaultdict(float))
                for curtag in self.unique_y:
                    for prevtag in self.unique_y:
                        weight = 0
                        for rexp in self.reg_exp:
                            try:
                                func_ind = self.func_filter_dict[rexp(x_j,curtag,prevtag,word_ind)]
                                param = self.lambdas[func_ind]
                                func = self.func_list[func_ind]
                                weight += math.exp(param*func(x_j,curtag,prevtag,word_ind))
                            except KeyError as e: pass
                        wdict[curtag][prevtag] = weight
                transition_dicts.append(wdict)
        return transition_dicts

    def viterbi_dict_search_crf(self,x_j,trans_dicts):
        #works on the less memory efficient dictionaries but clearer for debugging
        Q = []
        best_prev = []
        #Init Q
        for word_ind, word in enumerate(x_j):
            bp = {}
            QT = {}
            if word == self.START: pass
            elif word == self.END: pass
            elif word_ind == 1:
                for cur_tag in self.unique_y:
                    QT[cur_tag] = trans_dicts[word_ind][cur_tag][self.START]
            else:
                for curtag in trans_dicts[word_ind]:
                    best_score = 0
                    for prevtag_ind, prevtag in enumerate(trans_dicts[word_ind][curtag]):
                        r = trans_dicts[word_ind][curtag][prevtag] * Q[word_ind-1][prevtag]
                        if prevtag_ind == 0:
                            best_score = r
                            bp[curtag] = prevtag
                            QT[curtag] = r
                        elif r > best_score:
                            best_score = r
                            bp[curtag] = prevtag
                            QT[curtag] = r
            Q.append(QT)            
            best_prev.append(bp)
        best_prev.reverse()
        s = 0
        s_tag = ''
        QT = Q[-2]
        for curtag_ind, curtag in enumerate(QT):
            if curtag_ind == 0:
                s = QT[curtag]
                s_tag = curtag
            elif QT[curtag] > s:
                s = QT[curtag]
                s_tag = curtag
        ctag = s_tag
        tags = [s_tag]
        for BP in best_prev[1:-2]:
            ctag = BP[ctag]
            tags.insert(0,ctag)
        tags.insert(0,self.START)
        tags.append(self.END)
        return tags

    def sgd_crf(self,x,y):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.set_window_title('Training Accuracy versus Update Number')
        hl, = ax.plot([], [],'-k',label='black')
        num = 0
        for it in range(self.max_iter):
            print it
            accuracy = []
            order = range(len(y))
            shuffle(order)
            for exp_ind,j in enumerate(order):
                x_j = x[j]
                y_j = y[j]
                #Has a log term inside of the creation dict code
                matrices_j,funcs = self.create_matrices_crf(x_j)
                print x_j
                print y_j
                n = 0
                for ind,lamb in enumerate(self.lambdas):
                    self.lambdas[ind] = lamb - lamb*self.step_size*self.reg
                    n = n + lamb**2
                print 'Weight Norm:', n
                for func_ind in funcs:
                    func_i = self.func_list[func_ind]
                    param_i = self.lambdas[func_ind]
                    exp_val = self.expectation(x_j,y_j,matrices_j,func_i)
                    optimal_val = 0
                    for i in range(1,len(y_j)-1):
                        optimal_val += func_i(x_j,y_j[i],y_j[i-1],i)
                    print exp_val, optimal_val
                    self.lambdas[i] = param_i + self.step_size*(optimal_val - exp_val)
                
                score = (optimal_val - exp_val) / float(len(x_j))
                print score
                accuracy.append(score)
                self.update_line(hl,ax, num,score)
                num += 1
            self.step_size *= self.step_dec
            t_acc = sum(accuracy)/float(len(accuracy))
            yield num, self.step_size, it,t_acc                    
            
    def calc_Z(self,matrices):
        Z_mat = reduce(lambda x,y: np.dot(x,y),matrices)
        return Z_mat[0,-1]

    def expectation(self, x_j, y_j, matrices, func):
        alpha = np.zeros([len(self.unique_y),len(x_j)])
        beta = np.zeros([len(self.unique_y),len(x_j)])
        #Forward
        for i in range(len(x_j)):
            if y_j[i] == self.START:
                alpha[:,i] = 1
            else:
                #is this the right multiplication? Or should it be reversed?
                alpha[:,i] = np.dot(alpha[:,i-1],matrices[i])
        #Backward
        for i in range(len(x_j)-1,-1,-1):
            if y_j[i] == self.END:
                beta[:,i] = 1
            else:
                #is this the right multiplication? Or should it be reversed?
                beta[:,i] = np.dot(matrices[i+1],beta[:,i+1])
                
        Z = self.calc_Z(matrices)
        exp = 0
        p = 0
        for i in range(len(x_j)):
            for y1i,y1 in enumerate(self.unique_y):
                    for y2i,y2 in enumerate(self.unique_y):
                        prob = (alpha[y2i,i-1]*matrices[i][y1i,y2i]*beta[y1i,i])
                        prob_n = prob/Z
                        p += prob_n
                        exp += prob_n*func(x_j,y_j,y1i,y2i)
        return exp 

    def create_matrices_crf(self,x_j):
        matrices = []
        eligible_funcs = set()
        #for each token in the sequence
        for word_ind,word in enumerate(x_j):
            mat_i = np.zeros([len(self.unique_y),len(self.unique_y)])
            #cur tag
            for y1i,y1 in enumerate(self.unique_y):
                #prev tag
                for y2i,y2 in enumerate(self.unique_y):
                    weight = 0
                    for rexp in self.reg_exp:
                        try:
                            func_ind = self.func_filter_dict[rexp(x_j,y1,y2,word_ind)]
                            param = self.lambdas[func_ind]
                            func = self.func_list[func_ind]
                            val = func(x_j,y1,y2,word_ind)
                            if val != 0:
                                eligible_funcs.add(func_ind)
                            weight += param*val
                        except KeyError as e: pass
                        except IndexError as e: pass
                    mat_i[y1i,y2i] = math.exp(weight)
                matrices.append(mat_i)
        return matrices, eligible_funcs




    def create_fst_matrices(self,x_j):
        '''
        Gets a training sequence of inputs x_j which have been augmented with START and END so length is N+2
        @param x_j: the augmented input sequence
        @return: A sequence of N+1 matrices
        '''
        #previous on the rows
        #current on the columns
        #x_j is augmented with START and END
        N_2 = len(x_j)
        N_1 = len(x_j) - 1
        vocab_size = len(self.unique_y)
        #Holds 1XV, N (VxV), VX1
        matrices = []
        #Start with prev= START, cur = N[1]
        for i in range(1,N_1):
            #If it is the first sequence:
            if i == 1:
                cur_mat = []
                for ycur_ind,ycur in enumerate(self.unique_y):
                    weight = reduce(lambda prev,cur: prev + cur[0]*cur[1](x_j,ycur,self.START,i),
                                    zip(self.lambdas,self.func_list),0)
                    cur_mat.append(weight)
                cur_mat = np.array(cur_mat)
                '''
                #If it is the last sequence:
                elif i == len(x_j)-1:
                    cur_mat = []
                    for yprev_ind,yprev in enumerate(self.unique_y):
                        weight = reduce(lambda prev,cur: prev + cur[0]*cur[1](x_j,self.END,yprev,i),
                                        zip(self.lambdas,self.func_list),0)
                        cur_mat.append(weight)
                    cur_mat = np.array(cur_mat)  
                #If it is a normal sequence:
                '''
            else:
                cur_mat = np.zeros([vocab_size,vocab_size])
                for ycur_ind,ycur in enumerate(self.unique_y):
                    for yprev_ind,yprev in enumerate(self.unique_y):
                        weight = reduce(lambda prev,cur: prev + cur[0]*cur[1](x_j,ycur,yprev,i),
                                    zip(self.lambdas,self.func_list),0)
                        cur_mat[yprev_ind,ycur_ind] = weight
            matrices.append(cur_mat)
        return matrices
            
    def create_func_cache(self,x,):
        instance_results = []
        for x_j in x:
            function_results = []
            for func in self.func_list:
                edge_weights = {}
                for i in range(1,len(x_j)-1):
                    for ycur in self.unique_y:
                        for yprev in self.unique_y:
                            weight = func(x_j,ycur,yprev,i)
                            edge_weights[ycur+yprev] = weight
                function_results.append(edge_weights)
            instance_results.append(function_results)
        return instance_results     
        