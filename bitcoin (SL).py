import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

dataframe = pd.read_csv("C:\\Users\\Jenit\\Desktop\\bitcoin_price.csv")
print(dataframe)

print(dataframe.info())
print(dataframe.describe())

preprocess_dataframe = dataframe.dropna()
print(preprocess_dataframe.info())

print(preprocess_dataframe.isnull().values.any())

cols = preprocess_dataframe.columns[7]
X = preprocess_dataframe.drop(columns =cols)

cols = preprocess_dataframe.columns[0:7]
target_col = preprocess_dataframe.drop(columns =cols)
y = np.where(target_col.shift(-1) > target_col,1,-1).ravel()

scaler = MinMaxScaler(feature_range=(0, 1))
X= scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

print(X_train, "\n", X_test, "\n", Y_train, "\n", Y_test)

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, Y_train)
predict_linear_regression = linear_regression.predict(X_test)
error1 = round((mean_squared_error(predict_linear_regression, Y_test)), 2)
print("ACCURACY OF LINEAR REGRESSION : ", 100 - error1)

logistic_regression = linear_model.LogisticRegression()
logistic_regression.fit(X_train, Y_train)
predict_logistic_regression = logistic_regression.predict(X_test)
error2 = round((mean_squared_error(predict_logistic_regression, Y_test)), 2)
print("ACCURACY OF LOGISTIC REGRESSION : ", 100 - error2)

support_vector_machine = svm.LinearSVC()
support_vector_machine.fit(X_train, Y_train)
predict_support_vector_machine = support_vector_machine.predict(X_test)
error3 = round((mean_squared_error(predict_support_vector_machine, Y_test)), 2)
print("ACCURACY OF SUPPORT VECTOR MACHINE : ", 100 - error3)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, Y_train)
predict_naive_bayes = naive_bayes.predict(X_test)
error4 = round((mean_squared_error(predict_naive_bayes, Y_test)), 2)
print("ACCURACY OF MULTINOMIAL NAIVE BAYES : ", 100 - error4)

decision_tree = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
decision_tree.fit(X_train, Y_train)
predict_decision_tree = decision_tree.predict(X_test)
error5 = round((mean_squared_error(predict_decision_tree, Y_test)), 2)
print("ACCURACY OF DECISION TREE : ", 100 - error5)

lin_Acc = 100 - error1
log_Acc = 100 - error2
svm_Acc = 100 - error3
mnb_Acc = 100 - error4
dt_Acc = 100 - error5

# plot (width, height)
mp.figure(figsize = (12,7))
algorithm = ['Linear Regression', 'Logistic Regression', 'Support Vector Machine',
             'Multinomial Naive Bayes', 'Decision Tree']
accuracy = [lin_Acc, log_Acc, svm_Acc, mnb_Acc, dt_Acc]

# creating bar plot
mp.bar(algorithm, accuracy, width= 0.9, align='center', color='cyan', edgecolor = 'red')
# annotated text location
x = 1.0
y = 0.1

# bar plot annotation with accuracy
for x in range(len(algorithm)):
    mp.annotate(accuracy[x], (-0.1 + x, accuracy[x] + y))

# legend creation
mp.legend(labels = ['Accuracy'])

# plot title
mp.title("Comparison Study of Machine Learning Algorithms Based on Accuracy")

# x and y axis notations
mp.xlabel('MACHINE LEARNING ALGORITHMS')
mp.ylabel('PERFORMANCE')

# plot display
mp.show()




