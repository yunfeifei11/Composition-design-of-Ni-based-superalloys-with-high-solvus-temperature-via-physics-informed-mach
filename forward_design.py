import pickle
import pandas as pd
from physics_informed_featured_engineering import get_feature
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='inputpath is the path of the input alloy composition data file, such as data/example_data.csv. outputpath is the path to output the data file of predicted γ-plus solid solution temperature, such as data/example_result.csv')
parser.add_argument('inputpath', type=str,help='the path of the input alloy composition data')
parser.add_argument('outputpath', type=str,help='the data file of predicted γ-plus solid solution temperature')
args = parser.parse_args()
inputpath=args.inputpath
outputpath=args.outputpath

print("waiting")
f = open('model/physics_informed_model.pickle','rb')
rfc1 = pickle.load(f)
f.close()
feature=pd.read_csv("data/feature_data.csv")

data=pd.read_csv(inputpath)
ph_feature=get_feature(data,feature)
X=np.array(ph_feature.iloc[:,:-1])
result=rfc1.predict(X)
data.insert(0, 'predicted_values', result.tolist())
data.to_csv(outputpath,index=False)
print("calculated over.result saved "+outputpath)
