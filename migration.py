#!/Users/shchur/anaconda3/bin/python3

import sys
import collections
from scipy import (linalg,optimize)
from numpy import (dot,identity,mat)
import math
from math import (exp,log)

from CorrectLambda import CorrectLambda
from TwoPopulations import TwoPopulations
from OnePopulation import OnePopulation

class MigrationInference:
    def __init__(self):
        self.theta = 0.00001#coalescent mutation rate theta/2
        self.splitT = 3
        self.mu = [0.00001,0.00001]
        self.M = None
        self.integralP = None
        self.P0 = None
        self.P1 = None
        self.JAFS = [0 for i in range(7)]#0100,1100,0001,0101,1101,0011,0111
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
    
    def PrintMatrix(self):
        for i in range(self.Msize):
            for j in range(self.Msize):
                el = format(self.MM.item(i,j), '.10g')
                print( el, end = "\t" )
            print("")
            
    def PrintMatrixRow(self, rn):
        st = self.MapIndToState(rn)
        print (self.PrintState(st) )
        for i in range(self.Msize):
            el = format(self.MM.item(rn, i), '.10g')
            print( el, end = "\t" )
        print("")
        for i in range(self.Msize):
            if self.MM.item(rn, i) != 0:
                st = self.MapIndToState(i)
                print (self.PrintState(st) )
        
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
        model = TwoPopulations(self.lc[0][0], self.lc[0][1], self.mu[0], self.mu[1])
        self.P0 = [0.0 for i in range(model.Msize)]
        self.P0[2] = 1.0
        for interval in range(self.numT):
            print("Interval", interval)
            if interval < self.splitT:
                model = TwoPopulations(self.lc[interval][0], self.lc[interval][1], self.mu[0], self.mu[1])
            else:
                model = OnePopulation(self.lc[interval][0])
            if interval == self.splitT:
                self.CollapsePops()
            self.M = model.SetMatrix()
            print(self.M)
            self.SolveDifEq(interval)
            for i in range(model.Msize):
                jaf = model.StateToJAF(i)
                self.JAFS = [x + y*self.integralP[i] for x,y in zip(self.JAFS, jaf)]
            self.P0 = self.P1
    
    def CollapsePops(self):
        Pc = [0 for i in range(8)]
        Pc[0] = sum(self.P0[0:9])
        Pc[1] = sum(self.P0[9:15])
        Pc[2] = sum(self.P0[15:23])
        Pc[3] = sum(self.P0[23:29])
        Pc[4] = sum(self.P0[29:33])
        Pc[5] = sum(self.P0[33:37])
        Pc[6] = sum(self.P0[37:41])
        Pc[7] = sum(self.P0[41:44])
        self.P0 = Pc
    
    def SolveDifEq(self, interval):
        if interval < self.numT - 1:
            T = self.times[interval]
            MET = linalg.expm( dot(self.M,T) )
            self.P1 = dot(MET,self.P0)
        else:
            self.P1 = [0 for i in range( len(self.P0) )]
        MI = linalg.inv(self.M)
        sizeM = self.M.shape[0]
        self.integralP = [x - y for x, y in zip(self.P1, self.P0)]
        self.integralP = dot(MI,self.integralP)
    
    def ObjectiveFunction(self, mu):
        self.cl.SetMu(mu[0], mu[1])
        res = self.CorrectLambdas()
        if not res:
            return -10**(100)
        print(self.lc)
        self.JAFSpectrum()
        norm = sum(self.JAFS)
        print("----------",self.JAFS[0]/norm,self.JAFS[1]/norm,sep="\t\t")
        print(self.JAFS[2]/norm,self.JAFS[3]/norm,self.JAFS[4]/norm,sep="\t\t")
        print(self.JAFS[5]/norm,self.JAFS[6]/norm,"----------",sep="\t\t")
#        return self.Likelihood()
        return log(1)
    
    def Solve(self):
        self.cl = CorrectLambda()
        self.ObjectiveFunction([self.mu[0], self.mu[1]])
        #print(self.ObjectiveFunction([self.mu[0], self.mu[1]]))
#        optimize ObjectiveFunction(mu0, mu1)
    
    def Test(self):
        self.Solve()
        
Migration = MigrationInference()
Migration.Test()
