# step- 1 import the required libraries 

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import numpy as np

# step- 2  read the data 

data=pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
print(data.head())
print('the shape of our data {}'.format(data.shape))

#step 3 -  Describing our data

print('the description of our data \n  {}'.format(data.describe()))
 
#step 4 -Create a function to convert numerical values of the Index column to categorical values. 

def convert_index_to_name(ind):
    if ind==0 :
        return 'Extremely Weak'
    elif ind==1:
        return  ' Weak'
    elif ind==2 :
        return 'Normal'  
    elif ind==3:
        return 'Overweight' 
    elif ind==4:
        return 'Obesity'      
    elif ind==5:
        return 'Extreme Obesity'     
data['Index']=data['Index'].apply(convert_index_to_name)
print(data.head())

# step - 5 - visualize the data 
sns.lmplot('Height','Weight',data,hue='Index',size=7,aspect=1,fit_reg=False)
sns.lmplot('Height','Weight',data,hue='Gender',size=7,aspect=1,fit_reg=False)
plt.show()

# step -6- Analyze the value counts
print('the desturbution of data with respect to the Gender \n {}'.format(data['Gender'].value_counts()))
print('the desturbution of data with respect to the Index \n {}'.format(data['Index'].value_counts()))
print(data[data['Gender']=='Male']['Index'].value_counts())
print(data[data['Gender']=='Female']['Index'].value_counts())

#step 7- split the data 
y=data['Index']
X=data.drop(['Index'],axis=1)

#step-8 performe one hot encoding 
data_1=pd.get_dummies(X['Gender'])
X.drop(['Gender'],axis=1,inplace=True)
data=pd.concat([X,data_1],axis=1)

# step 9 : prepare the data 
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

# step 10 train the model

param_grid={'n_estimators':[100,200,300,400,500,600,700,800,1000],'max_features':[1,2,3,4]}
grid_search=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
grid_search.fit(X_train_scaled,y_train)
pred = grid_search.predict(X_test_scaled)
print(grid_search.best_params_)

print(classification_report(y_test,pred))
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print('Acuuracy is --> ',accuracy_score(y_test,pred)*100)
print('\n')

# step 11 live prediction 

def lp(details):
    gender = details[0]
    height = details[1]
    weight = details[2]
    
    if gender=='Male':
        details=np.array([[np.float(height),np.float(weight),0.0,1.0]])
    elif gender=='Female':
        details=np.array([[np.float(height),np.float(weight),1.0,0.0]])
    
    y_pred = grid_search.predict(scaler.transform(details))
    return (y_pred[0])
    
#Live predictor
your_details = ['Male',175,80]
print(lp(your_details))



