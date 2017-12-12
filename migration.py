#!/Users/shchur/anaconda3/bin/python3

import sys
import collections
from scipy import (linalg,optimize)
from numpy import (dot,identity,mat)
import math
from math import (exp,log)

from CorrectLambda import CorrectLambda

#lineage = collections.namedtuple('lineage', ['d0', 'd1', 'pop' ])#d0: number of descendents in population 0; d1: number of descendents in population 1; pop: current population of lineage

class lineage:
    def __init__(self, d0, d1, pop, fc = -1):
        self.d0 = d0
        self.d1 = d1
        self.pop = pop
        self.fc = fc

#class lineage(namedtuple('lineage', ['d0', 'd1', 'pop' , 'fc']):#d0: number of descendents in population 0; d1: number of descendents in population 1; pop: current population of lineage; fc - for lineage with three decendents is the sum of descendent populations in first coalescence
#    def __new__(cls, d0, d1, pop, fc=0):
#        return super(TemplateContainer, cls).__new__(cls, d0, d1, pop, fc)

class MigrationInference:
    def __init__(self):
        self.splitT = 3
        self.mu = [0.00001,0.00001]
        self.la = [0.1, 0.01]
        self.Msize = 52
        self.Pa = [0 for i in range(self.Msize)]
        self.Pa[2] = 1
#        self.P0 = mat(self.P0)
        self.P1 = [0 for i in range(self.Msize)]
#        self.P1 = mat([0 for i in range(self.Msize)])
        self.Pexpected = [0 for i in range(self.Msize)]
#        self.expP = mat([0 for i in range(self.Msize)])
        #self.expectP = [[0 for i in range(self.Msize)] for j in range(self.numT)]
        #PSMC parameters
        self.numT = 5 #number of time intervals
        self.lh = [[1,3],[0.1,0.5],[2,4],[1,0.5],[10,10]]#pairs of PSMC lambda_0 and lambda_1
        self.lc = [[0,0] for i in range(self.numT)]
        self.times = [2,3,1,5,10]
        #self.M - transition matrix
        #self.Minv - inversed transition matrix
        #self.Mexp - exponentiated matrix
        
    def CheckState(self, state):
        tn0, tn1 = 0, 0
        state[:] = sorted(state, key=lambda line: line.pop)
        state[:] = sorted(state, key=lambda line: line.d0, reverse = True)
        state[:] = sorted(state, key=lambda line: line.d0+line.d1, reverse=True)
        for val in state:
            tn0 += val.d0
            tn1 += val.d1
        if tn0 != 2 or tn1 != 2:
            print("CheckState() not passed: expected number of lineages is 2 and 2, instead there are", tn0, "and", tn1, "lineages")
            sys.exit(0)
        if len(state) == 2 and state[0].d0 + state[0].d1 == 3 and state[0].fc == -1:
            st = self.PrintState(state)
            print("CheckState() not passed: first coalescence for lineage with three descendants is not set:", st)
            sys.exit(0)
    
    def MapStateToInd(self, state):
        self.CheckState(state)
        if len(state) == 4:
            i, j = 0, 0
            for val in state:
                if val.d0 == 0:
                    i += val.pop
                else:
                    j += val.pop
            return i+3*j
        if len(state) == 3:
            if state[0].d0 + state[0].d1 != 2:
                print("Unexpected number of lineages, expected 2, recieved", state[0].d0 + state[0].d1)
                sys.exit(0)
            if state[0].d0 == 2:
                return 9 + 3*state[0].pop + state[1].pop + state[2].pop
            elif state[0].d0 == 1:
                return 9 + 6 + 4*state[0].pop + 2*state[1].pop + state[2].pop
            elif state[0].d0 == 0:
                return 9 + 6 + 8 + 3*state[0].pop + state[1].pop + state[2].pop
        if len(state) == 2:
            if state[0].d0 == 2 and state[0].d1 == 1:
                return 29 + 2*state[0].pop + state[1].pop + 4 * state[0].fc
            elif state[0].d0 == 1 and state[0].d1 == 2:
                return 29 + 8 + 2*state[0].pop + state[1].pop + 4 * (state[0].fc - 1)
            elif state[0].d0 == 2 and state[0].d1 == 0:
                return 29 + 8 + 8 + 2*state[0].pop + state[1].pop
            elif state[0].d0 == 1 and state[0].d1 == 1:
                return 29 + 8 + 8 + 4 + state[0].pop + state[1].pop
        return self.Msize
    
    def MapIndToState(self, ind):
        if not isinstance(ind, int) or ind < 0 or ind >= self.Msize:
            print("Unexpected index value", ind, ", index should be an integer between 0 and", self.Msize, ".")
            sys.exit(0)
        if ind < 9:
            state = [lineage(1, 0, 0), lineage(1, 0, 0), lineage(0, 1, 0), lineage(0, 1, 0)]
            i = ind//3
            j = ind%3
            while i > 0:
                self.UpdateLineagePop(state, i-1, 1)
                i -= 1
            while j > 0:
                self.UpdateLineagePop(state, 2+j-1, 1)
                j -= 1
        elif ind < 29:
            if ind < 15:
                state = [lineage(2, 0, 0), lineage(0, 1, 0), lineage(0, 1, 0)]
                self.UpdateLineagePop(state, 0, (ind - 9)//3)
                i = (ind - 9)%3
                while i > 0:
                    self.UpdateLineagePop(state, i, 1)
                    i -= 1
            elif ind < 23:
                state = [lineage(1, 1, 0), lineage(1, 0, 0), lineage(0, 1, 0)]
                self.UpdateLineagePop(state, 0, (ind - 15)//4)
                self.UpdateLineagePop(state, 1, ((ind - 15)%4)//2 )
                self.UpdateLineagePop(state, 2, ((ind - 15)%4)%2 )
            else:
                state = [lineage(0, 2, 0), lineage(1, 0, 0), lineage(1, 0, 0)]
                self.UpdateLineagePop(state, 0, (ind - 23)//3)
                i = (ind - 23)%3
                while i > 0:
                    self.UpdateLineagePop(state, i, 1)
                    i -= 1
        else:
            if ind < 37:
                state = [lineage(2, 1, 0), lineage(0, 1, 0)]
                popTmp = (ind - 29)%4
                self.UpdateLineagePop(state, 0, popTmp//2)
                self.UpdateLineagePop(state, 1, popTmp%2)
                self.UpdateLineageFC(state, 0, (ind - 29)//4)
            elif ind < 45:
                state = [lineage(1, 2, 0), lineage(1, 0, 0)]
                popTmp = (ind - 37)%4
                self.UpdateLineagePop(state, 0, popTmp//2)
                self.UpdateLineagePop(state, 1, popTmp%2)
                self.UpdateLineageFC(state, 0, (ind - 37)//4 + 1)
            elif ind < 49:
                state = [lineage(2, 0, 0), lineage(0, 2, 0)]
                self.UpdateLineagePop(state, 0, (ind - 45)//2)
                self.UpdateLineagePop(state, 1, (ind - 45)%2)
            else:
                state = [lineage(1, 1, 0), lineage(1, 1, 0)]
                if ind == 51:
                    self.UpdateLineagePop(state, 0, 1)
                if ind == 50 or ind == 51:
                    self.UpdateLineagePop(state, 1, 1)
        self.CheckState(state)
        return state
    
    def UpdateLineagePop(self, state, ind, pop):
        l = lineage(state[ind].d0, state[ind].d1, pop, state[ind].fc)
        state[ind] = l
    
    def UpdateLineageFC(self, state, ind, fc):
        l = lineage(state[ind].d0, state[ind].d1, state[ind].pop, fc)
        state[ind] = l
    
    def PrintState(self, state):
        st = ""
        for val in state:
            if val.d0 + val.d1 != 3:
                st = st + "(" + str(val.d0) + "," + str(val.d1) + "," + str(val.pop) + ") "
            else:
                if val.fc == 0:
                    fc = "0+0"
                elif val.fc == 1:
                    fc = "0+1"
                elif val.fc == 2:
                    fc = "1+1"
                else:
                    fc = str(val.fc) + " UNEXPECTED VALUE"
                st = st + "(" + str(val.d0) + "," + str(val.d1) + "," + str(val.pop) + "," + str(fc) + ") "
        return st
        
    def SetMatrix(self):
        self.MM = [[0 for col in range(self.Msize)] for row in range(self.Msize)]#transition matrix
        self.CM = [[0 for col in range(self.Msize)] for row in range(self.Msize)]#coalescence matrix
        for i in range(self.Msize):
            self.UpdateMatrixRow(i)
        self.MM = mat(self.MM)
        self.CM = mat(self.CM)
#        print self.MM
        
    def UpdateMatrixRow(self, ind):
        state = self.MapIndToState(ind)
        total = 0
        for i in range(len(state)):
            #Migration
            state1 = list(state)
            self.UpdateLineagePop(state1, i, (state[i].pop+1)%2)
            ind2 = self.MapStateToInd(state1)
            self.MM[ind][ind2] += self.mu[ state[i].pop ]
            total += self.mu[ state[i].pop ]
            #Coalescence
            for j in range(len(state)):
                if j == i or state[j].pop != state[i].pop:
                    continue
                state1 = list(state)
                for k in sorted([i,j], reverse=True):
                    del state1[k]
                line = lineage(state[i].d0+state[j].d0, state[i].d1+state[j].d1, state[i].pop)
                if line.d0 + line.d1 == 3:
                    if state[i].d0+state[i].d1 == 2:
                        line.fc = state[i].d1
                    else:
                        line.fc = state[j].d1
                state1.append(line)
                ind2 = self.MapStateToInd(state1)
                if ind2 != self.Msize:
                    self.MM[ind][ind2] += self.la[ state[i].pop ]
                    self.CM[ind][ind2] += self.la[ state[i].pop ]
                total += self.la[ state[i].pop ]/2.0
        self.MM[ind][ind] -= total
        
    def CorrectLambdas(self):
        p0 = [[1,0,0],[0,1,0]]
        for t in range(self.splitT):
#            print(self.lh[t])
#            print(self.times[t])
#            print(p0)
            self.cl.SetInterval(self.lh[t], self.times[t], p0)
            try:
                sol = self.cl.SolveLambdaSystem()
            except optimize.nonlin.NoConvergence:
                return False
            print("interval solution\t",sol)
            self.lc[t][0],self.lc[t][1] = sol[0][0],sol[0][1]
            p0 = sol[1]
        for t in range(self.splitT,self.numT):
            self.lc[t][0],self.lc[t][1] = (sol[0][0]+sol[0][1])/2,(sol[0][0]+sol[0][1])/2
        return True
            
    def JAFSpectrum(self):
        self.SetMatrix()
        print(self.MM)
        a = 1
    
    def ExpectationP(self, interval):
        if interval == 0:
            a = 0
        else:
            a = sum(self.times[0:interval-1])
        
        b = a + times[interval]
        T = b - a
        if interval == self.numT:
            MMET = [[0 for col in range(self.Msize)] for row in range(self.Msize)]#transition matrix
        else:
            MMET = linalg.expm( dot(self.M,T) )
        MMI = linalg.inv(self.MM)
        Pb = dot(MMET,self.Pa)
        Pe = [x - y for x, y in zip(Pb, Pa)]
        Pe = dot(MMI, Pe)
        Pe = [b*x - a*y  - z for x, y, z in zip(Pb, Pa, Pe)]
        Pe = dot(MMI,Pe)
        self.Pexpected = [x + y for x, y in zip(self.Pexpected, Pe)]
    
    def ObjectiveFunction(self, mu):
        self.cl.SetMu(mu[0], mu[1])
        res = self.CorrectLambdas()
        if not res:
            return -10**(100)
        print(self.lc)
        self.JAFSpectrum()
#        return self.Likelihood()
        return log(1)
    
    def Solve(self):
        self.cl = CorrectLambda()
        self.ObjectiveFunction([self.mu[0], self.mu[1]])
        #print(self.ObjectiveFunction([self.mu[0], self.mu[1]]))
#        optimize ObjectiveFunction(mu0, mu1)
    
    def Test(self):
        if 0:#Test if index-state-index mapping is identity
            for i in range(self.Msize):
                st = self.MapIndToState(i)
                print( self.MapStateToInd(st) )
        self.Solve()
    
    def Test1(self):
        self.SetMatrix()
        #self.MM = matrix(self.MM)
        m = []
        for i in range(9):
            m.append(self.MM[i][0:9])
        m = matrix(m)
        print(m)
        
Migration = MigrationInference()
Migration.Test()
