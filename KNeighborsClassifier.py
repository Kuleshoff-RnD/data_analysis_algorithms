# KNN Classification
import pandas
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
random_state = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = KNeighborsClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

#Выполнение примера выводит среднюю оценочную точность.
#0.726555023923
