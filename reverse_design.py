import pickle
from numpy import random
import geatpy as ea
import pandas as pd
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='inputpath is the path of the target γ-plus solves temperature/K, such as data/example_target_temperature.csv. outputpath is the path to output the data file of predicted alloy composition')
parser.add_argument('inputpath', type=str,help='the path of the input target γ-plus solves temperature/K')
parser.add_argument('outputpath', type=str,help='the data file of predicted alloy composition')
parser.add_argument('group', type=str,help='Number of alloy groups generated per target temperature')
args = parser.parse_args()
inputpath=args.inputpath
outputpath=args.outputpath
gr=args.group
f = open('model/rfc.pickle','rb')
rfc1 = pickle.load(f)
f.close()
regr=rfc1
x_len=29
def finall_grade_youxian(x):
    global regr
    global tar_tem
    x=np.array(x)
    result=regr.predict([x])[0]
    chengfa_sum=0
    weigth_chenfa1=15
    n_sum=16
    n_Ni=x[3]/100*n_sum##Ni
    if n_Ni>12:
        chengfa_sum=chengfa_sum+(n_Ni-12)*weigth_chenfa1
    
    
    n_Co=x[0]/100*n_sum##Co
    if n_Co>12:
        chengfa_sum=chengfa_sum+(n_Co-12)*weigth_chenfa1
    
    
    n_Al=x[1]/100*n_sum##Al
    if n_Al<1:
        chengfa_sum=chengfa_sum+(1-n_Al)*weigth_chenfa1
    if n_Al>2.5:
        chengfa_sum=chengfa_sum+(n_Al-2.5)*weigth_chenfa1
        
    n_Ti=x[4]/100*n_sum##Ti
    if n_Ti>1:
        chengfa_sum=chengfa_sum+(n_Ti-1)*weigth_chenfa1
    
    n_Nb=x[11]/100*n_sum##Nb
    if n_Nb>1:
        chengfa_sum=chengfa_sum+(n_Nb-1)*weigth_chenfa1
     
    n_Ta=x[7]/100*n_sum##Ta
    if n_Ta>1:
        chengfa_sum=chengfa_sum+(n_Ta-1)*weigth_chenfa1
    
    n_Cr=x[5]/100*n_sum##Cr
    if n_Cr>2:
        chengfa_sum=chengfa_sum+(n_Cr-2)*weigth_chenfa1
    
    n_Mo=x[9]/100*n_sum##Mo
    if n_Mo>1:
        chengfa_sum=chengfa_sum+(n_Mo-1)*weigth_chenfa1
    
    n_W=x[2]/100*n_sum##W
    if n_W>1:
        chengfa_sum=chengfa_sum+(n_W-1)*weigth_chenfa1
    
    chenfen_sum=100
    for i in range(len(x)):
        chenfen_sum=chenfen_sum-x[i]
    chenfen_sum=np.abs(chenfen_sum)
    if chenfen_sum>5:
         chengfa_sum=chengfa_sum+100*(chenfen_sum-5)
    return 1/(np.abs(result-tar_tem+np.random.rand()*20)+1e-5)
def aimFunc(pop):  
    Vars = pop.Phen
    
    
    cv=Vars[:,[0]]*0
    d=np.zeros(shape=(len(Vars),1))
    for i in range(len(Vars)):
        d[i]=finall_grade_youxian(Vars[i])
    pop.ObjV = d
    pop.CV = cv
class MyProblem(ea.Problem):  
    global x_len
    def __init__(self):
        b=[]
        c=[]
        for i in range(x_len):
            b.append(0)
            c.append(100)
        l_lb = b 
        u_ub = c  
        name = 'MyProblem'  
        M = 1  
        maxormins = [-1]  
        Dim = x_len  
        varTypes = [0] * Dim
        lb = l_lb  
        ub = u_ub  
        lbin = [1] * Dim   
        ubin = [1] * Dim         
        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)

    def evalVars(self, Vars): 
        cv=Vars[:,[0]]*0
        d=np.zeros(shape=(len(Vars),1))
        for i in range(len(Vars)):
            d[i]=finall_grade_youxian(Vars[i])
        f=d
        CV=cv
        return f, CV                 

    def calReferObjV(self):  
        referenceObjV = np.array([[0]])
        return referenceObjV  
    
#group=2
group=int(gr)
#df=pd.read_csv("../data/example_target_temperature.csv")
df=pd.read_csv(inputpath)
target=[]
for i in df.iloc[:,0]:
    for k in range(group):
        target.append(i)
target=np.array(target)  
#output_file="reverse_Y_ele_lim3_test.txt"
output_file=outputpath
os.system("rm "+output_file)
os.system("touch "+output_file)
count=0
ele_str="Co,Al,W,Ni,Ti,Cr,Ge,Ta,B,Mo,Re,Nb,Mn,Si,V,Fe,Zr,Hf,Ru,Ir,La,Y,Mg,C,Cu,P,S,Pt,Ar"
os.system("echo '"+"number"+" ,"+"target"+" ,"+"predicted"+", "+ele_str+"' >> "+output_file)
for tar in target:
    count=count+1
    tar_tem=tar
    problem = MyProblem()  
    algorithm = ea.soea_DE_currentToBest_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=200),
        MAXGEN=40,  
        logTras=5)  
    algorithm.mutOper.F = 0.7  
    algorithm.recOper.XOVR = 0.7  
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True)
    a=""
    for i in range(len(res['Vars'][0])):
        a=a+str(res['Vars'][0][i])+" "
    x=res['Vars'][0]
    x=np.array(x)
    result=regr.predict([x])[0]
    os.system("echo '"+str(count)+" "+str(tar_tem)+" "+str(result)+" "+a+"' >> "+output_file)