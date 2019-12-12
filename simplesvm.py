import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def classify_SVM(df):
    targets= df['bandgaps']
    features= df.drop('ids', axis=1)
    features= features.drop('bandgaps', axis=1)
    features= features.drop('coulomb_original', axis=1)
    features= features.drop('coulomb_padded', axis=1)
    features= features.drop('eig', axis=1)
    features= features.drop('_symmetry_space_group_name_H-M', axis=1)
    features= features.drop('_chemical_formula_sum', axis=1)

    #test set size is 20% of dataset because that's what the authors did
    x_train, x_test, y_train, y_test= train_test_split(list(df['eig']), targets, test_size=.20, random_state=42)
    newgamma=1/(10000*1000)
    clf= svm.SVR(gamma=newgamma)
    #print(len(x_train))
    #print(len(y_train))
    clf.fit(x_train,y_train)
    predictions= clf.predict(x_test)
    print(predictions)
    print(y_test)

    mse = (np.square(y_test - predictions)).mean()
    print(mse)
    print(np.sqrt(mse))
    mae= np.abs(y_test-predictions).mean()
    print(mae)

def linreg(df):
    targets= df['bandgaps']
    features= df.drop('ids', axis=1)
    features= features.drop('bandgaps', axis=1)
    features= features.drop('coulomb_original', axis=1)
    features= features.drop('coulomb_padded', axis=1)
    features= features.drop('eig', axis=1)
    features= features.drop('_symmetry_space_group_name_H-M', axis=1)
    features= features.drop('_chemical_formula_sum', axis=1)

    #test set size is 20% of dataset because that's what the authors did
    x_train, x_test, y_train, y_test= train_test_split(features, targets, test_size=.20, random_state=42)
    reg = LinearRegression().fit(x_train,y_train)
    predictions= reg.predict(x_test)

    mse = (np.square(y_test - predictions)).mean()
    print(mse)
    print(np.sqrt(mse))
    mae= np.abs(y_test-predictions).mean()
    print(mae)

def list_from_string(sadstring):
    sadstring=sadstring[1:-1]
    sadlist=sadstring.split(', ')
    happylist=[]
    for x in sadlist:
        happylist.append(float(x))
    return happylist

df=pd.read_csv('fulldf.csv',encoding = 'unicode_escape')

sorted_eigs = df['eig']
converted_eigs = []
for val in sorted_eigs:
    temp = []
    eig_str = val[1:-1] # get rid of brackets
    try:
        temp = [float(x) for x in eig_str.split(',')]
    except:
        t = [complex(''.join(a.split())) for a in eig_str.split(',')]
        temp = [z.real for z in t]
    converted_eigs.append(temp)
df['eig']=converted_eigs
#print(converted_eigs[0:2])
df['_chemical_formula_weight']=df['_chemical_formula_weight'].apply(float)
df=df.dropna(axis=0,how='any')

# encoding space group
space_vals = df["_symmetry_space_group_name_H-M"]
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(space_vals)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
space_vals_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(space_vals_encoded[0:3])
temp=space_vals_encoded[0]
for x in range(len(temp)):
    e='encode'+str(x)
    df[e]=space_vals_encoded[:,x]
tempeig=list(df['eig'])

for x in range(208):
    e='eig'+str(x)
    df[e]=[row[x] for row in tempeig]

#print(df['eig0'])
linreg(df)
