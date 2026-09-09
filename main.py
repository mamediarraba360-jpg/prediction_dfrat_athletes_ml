import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
draft=pd.read_csv("dfat.csv")
print(draft.head())
print(draft.info())
print(draft.isnull().sum())
draft_count=draft["Drafted"].value_counts()
print(draft_count)
draft_percentage=draft["Drafted"].value_counts(normalize=True*100)
print(draft_percentage)
draft_numeric=draft.select_dtypes(include="number")
draft.drop(["Id","Drafted"],axis=1)
print(draft_numeric.corr())
print(draft.describe())
corr_target=draft_numeric.corr()["Drafted"].sort_values(ascending=False)
print(corr_target)
print(draft_numeric.drop(columns=["Drafted"]).corr())
info=draft[['Id']].copy()
draft.drop(columns=['Id'],inplace=True)
draft_School=draft['School'].copy()
from sklearn.preprocessing import LabelEncoder
label_encoders={}
for c in ["Player_Type", "Position_Type", "Position","School"]:
    label_encoders[c] = LabelEncoder()
    draft[c] = label_encoders[c].fit_transform(draft[c].astype(str))
draft_mean=(["Age","Sprint_40yd","Vertical_Jump","Shuttle","Agility_3cone","Broad_Jump"
,"Bench_Press_Reps"])
draft[draft_mean]=draft[draft_mean].fillna(draft[draft_mean].mean())
print(draft.isnull().sum())
x=draft.drop(columns=["Drafted"])
y=draft["Drafted"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn .tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred=model.predict(x_test)
cm=metrics.confusion_matrix(y_test,y_pred)
dispo=ConfusionMatrixDisplay(confusion_matrix=cm)
dispo.plot()
plt.show()
y_pred=model.predict(x_test)
precision=metrics.precision_score(y_test,y_pred)
recall=metrics.recall_score(y_test,y_pred)
print("Recall:",recall)
print("precision:",precision)
draft=x_test.copy()
draft['Drafted']=y_test.values
draft["y_pred"]=y_pred
draft['Id']=info.loc[draft.index,'Id']
draft['School']=draft_School.loc[draft.index]
draft.to_csv("NFL.csv",index=False)