'''
Created on Oct 28, 2013

@author: samshuster
'''
from random import shuffle
from random import seed
import numpy as np
import math
import features
import cPickle as pickle

class CRF(object):
    '''
    Is a Conditional Random Field object upon which viterbi and forward-backward algorithm can be run
    '''
    START = 'START'
    END = 'END'

    def __init__(self, func_list = [], step_size = 1, reg = 0.1,max_iter=3):
        '''
        Create a CRF from a dictionary
        '''
        seed(1)
        self.func_list = func_list
        self.step_size = step_size
        self.reg = reg
        self.max_iter = max_iter
    
    def fit(self, x, y):
        '''
        @param x: A list of a sequence of inputs (ordered, list of list)
        @param y: A list of a sequence of outputs (ordered, list of list)
        @param func_list: A list of functions that take two inputs, a list of inputs, a list of outputs
        '''
        self.preprocess_data(x,y)
        self.infer_features(x, y)
        self.lambdas = [0 for _ in self.func_list]
        self.sgd(x,y)
    
    def save(self,out_file):
        out = [self.lambdas,self.unique_y]
        pickle.dump(out,open(out_file,'wb'))
       
    def load(self,out_file, x, y):
        [self.lambdas,self.unique_y] = pickle.load(open(out_file,'rb'))
        self.preprocess_data(x,y)
        self.infer_features(x,y)
        
    def predict(self, x):
        for x_j in x:
            matrices_j = self.create_matrices(x_j)
            self.viterbi_search(x_j,matrices_j)
            
    def viterbi_search(self,x_j,trans_matrices):
        Q = np.zeros([len(self.unique_y),len(x_j)])
        best_pred = np.zeros([len(self.unique_y),len(x_j)])
        Q[:,0] = trans_matrices[1][:,0]
        for i in range(2,len(x_j)):
            for j in range(len(self.unique_y)):
                Q[j,i] = 0
                best_pred[j,i] = 0
                best_score = float('-inf')
                for k in range(len(self.unique_y)):
                    r = trans_matrices[i][j,k]*Q[k,i-1]
                    if r > best_score:
                        best_score = r
                        best_pred[j,i] = k
                        Q[j,i] = r
        final_best = 0
        final_score = float('-inf')
        for j in range(len(self.unique_y)):
            if Q[j,len(x_j)-1] > final_score:
                final_score = Q[j,len(x_j)-1]
                final_best = j
        print self.unique_y[final_best]
        current = final_best
        for i in range(len(x_j)-2,-1,-1):
            current = best_pred[int(current),i+1]
            print self.unique_y[int(current)]
    
    def infer_features(self, x, y):
        self.feat_labels = []
        for x_j in self.unique_x:
            for y_j in self.unique_y:
                new_func = lambda words,curr_tag,prev_tag,i,y_j=y_j,x_j=x_j : features.atomic_tag_label(words, curr_tag, prev_tag, i, tag=y_j, word=x_j)
                self.func_list.append(new_func)
                self.feat_labels.append((x_j,y_j))
        #self.func_list = self.func_list[:20]
        #self.feat_labels = self.feat_labels[:20]
        for y1 in self.unique_y:
            for y2 in self.unique_y:
                new_func = lambda words,curr_tag,prev_tag,i,y1=y1,y2=y2 : features.bigram_tag_label(words, curr_tag, prev_tag, i, tag=y1,tag2=y2)
                self.func_list.append(new_func)
                self.feat_labels.append((y1,y2))
        #self.func_list = self.func_list[-20:]
        #self.feat_labels = self.feat_labels[-20:]
        self.func_list = self.func_list[-200:-195]
        self.feat_labels = self.feat_labels[-200:-195]
        
    def preprocess_data(self, x, y):
        '''Generates features and returns a list of dictionaries'''
        self.unique_x = set()
        self.unique_y = set()
        for x_j,y_j in zip(x,y):
            self.unique_x = self.unique_x.union(set(_ for _ in x_j))
            self.unique_y = self.unique_y.union(set(_ for _ in y_j))
            x_j.insert(0,self.START)
            y_j.insert(0,self.START)  
            x_j.append(self.END)
            y_j.append(self.END)
        self.unique_x = list(self.unique_x)
        self.unique_y = list(self.unique_y)
        self.unique_y.insert(0,self.START)
        self.unique_y.append(self.END)
        
    def sgd_crf(self,x,y):
        for it in range(self.max_iter):
            print it
            for i in range(len(self.func_list)):
                func_i = self.func_list[i]
                order = range(len(y))
                shuffle(order)
                for exp_ind,j in enumerate(order):
                    param_i = self.lambdas[i]
                    reg_term = self.step_size*self.reg*param_i
                    x_j = x[j]
                    y_j = y[j]
                    if exp_ind == 0:
                        matrices_j = self.create_matrices(x_j)
                        exp = self.expectation(x_j,y_j, matrices_j, func_i)
                    cur_value = 0
                    for position in range(1,len(x_j)):
                        cur_value += func_i(x_j,y_j[position],y_j[position-1],position)
                    self.lambdas[i] = param_i - reg_term + self.step_size*(cur_value - exp)
                    print cur_value, reg_term, self.lambdas[i], exp
            print self.lambdas
            print self.feat_labels
                    
    
    def create_matrices(self,x_j):
        matrices = []
        #for each token in the sequence
        for i in range(0,len(x_j)):
            mat_i = np.zeros([len(self.unique_y),len(self.unique_y)])
            #cur tag
            for y1i,y1 in enumerate(self.unique_y):
                #prev tag
                for y2i,y2 in enumerate(self.unique_y):
                    s = 0
                    for lamb_j,func_j in zip(self.lambdas,self.func_list):
                        s+= lamb_j * func_j(x_j,y1,y2,i)
                    mat_i[y1i,y2i] = math.exp(s)
            matrices.append(mat_i)
        return matrices
    
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
                        if prob_n > 1:
                            print prob, prob_n
                        p += prob
                        exp += prob*func(x_j,y_j,y1i,y2i)
        print p
        exit()
        return exp 
            
    
    