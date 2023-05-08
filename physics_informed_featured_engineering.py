import pandas as pd 
import numpy as np
def get_ele_pro(feature,ele,pro):
    result = feature.loc[feature['Formula'] == ele, pro]
    return float(result)
def get_max_pro(data,feature,pro):
    value=[]
    for ele in data.index:
        value.append(get_ele_pro(feature,ele,pro)*data[ele]*0.01)
    result=np.max(value)
    return float(result)
def get_min_pro(data,feature,pro):
    value=[]
    for ele in data.index:
        value.append(get_ele_pro(feature,ele,pro)*data[ele]*0.01)
    result=np.min(value)
    return float(result)
def get_range_pro(data,feature,pro):
    return get_max_pro(data,feature,pro)-get_min_pro(data,feature,pro)
def get_reduce_pro(data,feature,pro):
    re=get_ele_pro(feature,data.index[0],pro)*data[data.index[0]]*0.01
    flag=False
    for ele in data.index:
        if flag==False:
            flag=True
        else:
            re=(re*get_ele_pro(feature,data.index[0],pro)*data[data.index[0]]*0.01)/(re+get_ele_pro(feature,data.index[0],pro)*data[data.index[0]]*0.01+1e-6)
    if re == 0:
        result=-1
    else:
        result=re
    return result
def get_feature(data_all,feature):
    Max_UP=[]
    range_UP=[]
    Max_VP=[]
    range_VP=[]
    reduce_VD=[]
    reduce_UD=[]
    range_SG=[]
    Min_VD=[]
    Min_MM=[]
    range_PC=[]
    for i in range(len(data_all)):
        data=data_all.iloc[i,:]
        Max_UP.append(get_max_pro(data,feature,"UP"))
        range_UP.append(get_range_pro(data,feature,"UP"))
        Max_VP.append(get_max_pro(data,feature,"VP"))
        range_VP.append(get_range_pro(data,feature,"VP"))
        reduce_VD.append(get_reduce_pro(data,feature,"VD"))
        reduce_UD.append(get_reduce_pro(data,feature,"UD"))
        range_SG.append(get_range_pro(data,feature,"SG"))
        Min_VD.append(get_min_pro(data,feature,"VD"))
        Min_MM.append(get_min_pro(data,feature,"MM"))
        range_PC.append(get_range_pro(data,feature,"PC"))
    result= pd.DataFrame()  
    result["Max_UP"]=Max_UP
    result["range_UP"]=range_UP
    result["Max_VP"]=Max_VP
    result["range_VP"]=range_VP
    result["reduce_VD"]=reduce_VD
    result["reduce_UD"]=reduce_UD
    result["range_SG"]=range_SG  
    result["Min_VD"]=Min_VD
    result["Min_MM"]=Min_MM
    result["range_PC"]=range_PC
    for i in data_all.columns:
        result[i]=data_all[i]
    return result