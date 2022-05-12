import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib
import joblib
from sklearn import tree
df = pd.read_csv('assets/music.csv')
# print(type(df))
# print(type(df['Year'].values))
# print(df['Year'].values)
# print(df.values)
X=df.drop(columns=['genre'])
Y=df['genre']
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)
# print((X , Y))
model = DecisionTreeClassifier()
model.fit(X_train , Y_train)

# Visulizaiton a Decision tree
tree.export_graphviz(model , out_file='music-recommender.dot' , 
feature_names=['age' ,'gender'] , class_names=sorted(Y.unique())
, label='all' , rounded=True , filled=True)



# model persistent
joblib.dump(model ,'music-recommender.joblib')
obj=joblib.load('music-recommender.joblib')
obj.predict([[1 ,1]])


# prediction = model.predict(X_test)
# acc_score=accuracy_score(Y_test , prediction)
# print("acc_score::" , acc_score)