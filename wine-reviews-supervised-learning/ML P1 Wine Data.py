
# coding: utf-8

# In[114]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib
from matplotlib import pyplot as plt
import keras
import warnings
from nltk.stem import PorterStemmer
from nltk import word_tokenize, sent_tokenize


from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import validation_curve
import sklearn_evaluation 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# In[115]:


wine_red = pd.read_csv('winequality-red.csv',sep=';')
wine_white = pd.read_csv('winequality-white.csv',sep=';')
wine = pd.concat([wine_red, wine_white])
wine.head()


# In[116]:


wine['quality'][wine['quality']<7] = 0
wine['quality'][wine['quality']>=7] = 1


# In[117]:


wine.head()


# In[118]:


wine['quality'].value_counts()


# In[119]:


wine['fixed acidity'].hist()


# In[120]:


wine['citric acid'].hist()


# In[121]:


wine_quality = wine['quality']
wine_quality.hist()
del wine['quality']


# In[122]:


wine.describe()


# In[123]:


wine.head()


# In[124]:


wine.shape


# In[125]:


x_train, x_test, y_train, y_test = train_test_split(wine, wine_quality, test_size=0.3, random_state=42)
scaler = StandardScaler()
scaler.fit(wine.values)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[126]:


print("Train Rows:" + str(x_train.shape))
print("Train Labels:" + str(y_train.shape))
print("Test Rows:" + str(x_test.shape))
print("Test Labels" + str(y_test.shape))


# In[127]:


y_train.hist()


# In[128]:


y_test.hist()


# In[129]:


#baseline
import random
baseline = random.choices(population=[0,1],weights=[0.87, 0.13],k=x_train.shape[0])
print("Baseline Accuracy:" + str(sklearn.metrics.accuracy_score(y_train, baseline)))
print("Baseline Precision:" + str(sklearn.metrics.precision_score(y_train, baseline)))
print("Baseline Recall:" + str(sklearn.metrics.recall_score(y_train, baseline)))
print("Baseline F1 Score:" + str(sklearn.metrics.f1_score(y_train, baseline)))


# In[130]:


def plot_model_complexity(grid_search_results, param_name):
    param = 'param_'+str(param_name)
    means_train = grid_search_results.groupby([param])['mean_train_score'].mean().reset_index()
    means_cv = grid_search_results.groupby([param])['mean_test_score'].mean().reset_index()
    #means_train.sort_values(by=[param], inplace=True)
    #means_cv.sort_values(by=[param], inplace=True)
    #means_train[param] = means_train[param].astype(str)
    #means_cv[param] = means_cv[param].astype(str)
    
    param_values = list(means_train[param])
    #param_values = ['val: ' + str(i) for i in param_values]
    
    plt.figure(figsize=(8, 8))
    plt.plot(param_values, means_train['mean_train_score'], 'r')
    plt.plot(param_values, means_cv['mean_test_score'], 'b')
    plt.title(param_name + " vs F1 Score")
    plt.xlabel(param_name)
    plt.ylabel('F1 Score')
    plt.ylim(means_cv['mean_test_score'].min()-0.1)
    plt.legend()
    plt.show()


# In[131]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# In[132]:


#Decisison Tree Grid Search
dt_parameters = {'min_samples_leaf':[1,5, 10, 20, 25,50]}
dt = DecisionTreeClassifier()
dt_clf = GridSearchCV(dt, dt_parameters,cv=cv, scoring = 'f1')
dt_clf.fit(x_train, y_train)
dt_grid_search_results = pd.DataFrame(dt_clf.cv_results_).sort_values(by='rank_test_score')


# In[133]:


dt_grid_search_results = pd.DataFrame(dt_clf.cv_results_).sort_values(by='rank_test_score')


# In[134]:


plot_model_complexity(dt_grid_search_results, 'min_samples_leaf')


# In[135]:


#Decisison Tree Grid Search
dt_parameters = {'max_depth':[1, 5, 10, 20, 25,50]}
dt = DecisionTreeClassifier()
dt_clf = GridSearchCV(dt, dt_parameters,cv=cv, scoring = 'f1')
dt_clf.fit(x_train, y_train)
dt_grid_search_results = pd.DataFrame(dt_clf.cv_results_).sort_values(by='rank_test_score')


# In[136]:


plot_model_complexity(dt_grid_search_results, 'max_depth')


# In[137]:


#Decisison Tree Grid Search
dt_parameters = {'max_depth':[5,10,15,25,50], 'min_samples_leaf':[1, 5, 10, 20, 25, 50]}
dt = DecisionTreeClassifier()
dt_clf = GridSearchCV(dt, dt_parameters,cv=cv, scoring = 'f1')
dt_clf.fit(x_train, y_train)
dt_grid_search_results = pd.DataFrame(dt_clf.cv_results_).sort_values(by='rank_test_score')

# View the best parameters for the model found using grid search
print('Best score:', dt_clf.best_score_) 
print('Best Max Depth:',dt_clf.best_estimator_.max_depth) 
print('Best Min Leaf Samples:',dt_clf.best_estimator_.min_samples_leaf)


# In[138]:


#SVM C Search
parameters = {'C':[0.001,0.01,0.1,1,10]}
svm = SVC(verbose =0, kernel='rbf')
svm_clf = GridSearchCV(svm, parameters, cv= cv, scoring = 'f1')
svm_clf.fit(x_train, y_train)
svm_grid_search_results = pd.DataFrame(svm_clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(svm_grid_search_results, 'C')


# In[254]:


#SVM C Search
parameters = {'C':[0.001,0.01,0.1,1,10]}
svm = SVC(verbose =0, kernel='linear')
svm_clf = GridSearchCV(svm, parameters, cv= cv, scoring = 'f1')
svm_clf.fit(x_train, y_train)
svm_grid_search_results = pd.DataFrame(svm_clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(svm_grid_search_results, 'C')


# In[139]:


#SVM gamma Search
parameters = {'gamma':[0.0001, 0.001, 0.01, 0.1, 1]}
svm = SVC(verbose =0, kernel='rbf')
svm_clf = GridSearchCV(svm, parameters, cv= cv, scoring = 'f1')
svm_clf.fit(x_train, y_train)
svm_grid_search_results = pd.DataFrame(svm_clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(svm_grid_search_results, 'gamma')


# In[140]:


#SVM Grid Search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
parameters = {'C':[0.1,1,5,10], 'gamma':[0.01, 0.1, 0.2, 0.3], 'kernel':['rbf','linear']}
svm = SVC(verbose=2)
svm_clf = GridSearchCV(svm, parameters, cv= cv, scoring = 'f1')
svm_clf.fit(x_train, y_train)

svm_grid_search_results = pd.DataFrame(svm_clf.cv_results_).sort_values(by='rank_test_score')
# View the best parameters for the model found using grid search
print('Best score for data1:', svm_clf.best_score_) 
print('Best Estimators:',svm_clf.best_estimator_.gamma) 
print('Best C:',svm_clf.best_estimator_.C)
print('Best Kernel:',svm_clf.best_estimator_.kernel)


# In[141]:


#GBM Grid Search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
parameters = {'learning_rate':[0.001, 0.01, 0.1, 0.3, 0.5], 'n_estimators':[10,100,250,500,1000], 'max_depth':[1,3,5,7,10]}
gbm = GradientBoostingClassifier(verbose =1)
gbm_clf = GridSearchCV(gbm, parameters, cv= cv, scoring = 'f1')
gbm_clf.fit(x_train, y_train)


# In[142]:


gbm_grid_search_results = pd.DataFrame(gbm_clf.cv_results_).sort_values(by='rank_test_score')
# View the best parameters for the model found using grid search
print('Best score for data1:', gbm_clf.best_score_) 
print('Best Estimators:',gbm_clf.best_estimator_.n_estimators) 
print('Best Shrinkage:',gbm_clf.best_estimator_.learning_rate)
print('Best Depth:',gbm_clf.best_estimator_.max_depth)


# In[143]:


plot_model_complexity(gbm_grid_search_results, 'learning_rate')


# In[144]:


plot_model_complexity(gbm_grid_search_results, 'n_estimators')


# In[249]:


plot_model_complexity(gbm_grid_search_results, 'max_depth')


# In[145]:


#KNN Grid Search
parameters = {'n_neighbors':[1,3,5,7,10,25]}
knn = KNeighborsClassifier()
knn_clf = GridSearchCV(knn, parameters, cv= 5, scoring = 'f1')
knn_clf.fit(x_train, y_train)


# In[146]:


knn_grid_search_results = pd.DataFrame(knn_clf.cv_results_).sort_values(by='rank_test_score')
# View the best parameters for the model found using grid search
print('Best score for data1:', knn_clf.best_score_) 
print('Best K:',knn_clf.best_estimator_.n_neighbors) 


# In[147]:


plot_model_complexity(knn_grid_search_results, 'n_neighbors')


# In[250]:


#Neural Network Grid Search
parameters = {'learning_rate_init':[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3]}
nn = MLPClassifier(activation='relu', solver='adam',verbose = 1)
clf = GridSearchCV(nn, parameters, cv= 3, scoring = 'f1')
clf.fit(x_train, y_train)
nn_grid_search_results = pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(nn_grid_search_results, 'learning_rate_init')


# In[149]:


#Neural Network Grid Search
parameters = {'hidden_layer_sizes':[8,16,32,64,128,256,512]}
nn = MLPClassifier(activation='relu', solver='lbfgs')
clf = GridSearchCV(nn, parameters, cv= 3, scoring = 'f1')
clf.fit(x_train, y_train)
nn_grid_search_results = pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(nn_grid_search_results, 'hidden_layer_sizes')


# In[150]:


#Neural Network Grid Search
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01], 'hidden_layer_sizes':[8,16,32,64,128]
              ,'learning_rate_init':[0.00000001, 0.00001, 0.0001, 0.001, 0.01],'activation':['relu','logistic']
            , 'solver':['adam','lbfgs']}
nn = MLPClassifier(verbose =1)
nn_clf = GridSearchCV(nn, parameters, cv= cv, scoring = 'f1')
nn_clf.fit(x_train, y_train)

nn_grid_search_results = pd.DataFrame(nn_clf.cv_results_).sort_values(by='rank_test_score')

print('Best score:', nn_clf.best_score_) 
print('Best Estimators:',nn_clf.best_estimator_.alpha) 
print('Best Learning Rate:',nn_clf.best_estimator_.learning_rate)
print('Best HL:',nn_clf.best_estimator_.hidden_layer_sizes)
print('Best Activation:',nn_clf.best_estimator_.activation)
print('Best Optimizer:',nn_clf.best_estimator_.solver)


# In[151]:


print('Best score:', nn_clf.best_score_) 
print('Best Alpha:',nn_clf.best_estimator_.alpha) 
print('Best Learning Rate:',nn_clf.best_estimator_.learning_rate_init)
print('Best Hidden Layer:',nn_clf.best_estimator_.hidden_layer_sizes)
print('Best Activation:',nn_clf.best_estimator_.activation)
print('Best Activation:',nn_clf.best_estimator_.solver)


# In[153]:


#http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring ='f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[154]:


print(svm_clf.best_estimator_)


# In[155]:


# SVC Learning Curve
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
new_svm = SVC(verbose =1, kernel='rbf')
plot_learning_curve(svm_clf.best_estimator_, "Learning Curve for SVM Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=4)
plt.show()


# In[156]:


#DT Learning Curve
plot_learning_curve(dt_clf.best_estimator_, "Learning Curves forDT Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=4)
plt.show()


# In[157]:


#GBM Learning Curve
plot_learning_curve(gbm_clf.best_estimator_, "Learning Curve for GBM Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=4)
plt.show()


# In[158]:


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
plot_learning_curve(MLPClassifier(alpha=0.00001,learning_rate_init=0.01,hidden_layer_sizes=128
                                  ,activation='relu',solver='lbfgs',verbose=True)
                                  ,"Learning Curve for NN Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=1)
plt.show()


# In[160]:


#KNN Learning Curve
plot_learning_curve(knn_clf.best_estimator_, "Learning Curve for Best KNN Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=4)
plt.show()


# In[242]:


##EVALUATION ON TEST DATA
dt_preds = dt_clf.predict(x_test)
nn_preds = nn_clf.predict(x_test)
knn_preds = knn_clf.predict(x_test)
gbm_preds = gbm_clf.predict(x_test)
svm_preds = svm_clf.predict(x_test)

def get_performance(model, y_test, preds):
    f = round(f1_score(y_test, preds),3)
    a = round(accuracy_score(y_test, preds),3)
    p = round(precision_score(y_test, preds),3)
    r = round(recall_score(y_test, preds),3)
    
    return [model, f,a,p,r]

test_results =[] 
test_results.append(['Baseline', round(f1_score(y_train, baseline),3), round(accuracy_score(y_train, baseline),3)
                     , round(precision_score(y_train, baseline),3), round(recall_score(y_train, baseline),3)])
test_results.append(get_performance('Decision Tree' , np.array(y_test), dt_preds))
test_results.append(get_performance('SVM', np.array(y_test), svm_preds))
test_results.append(get_performance('GBM', np.array(y_test), gbm_preds))
test_results.append(get_performance('KNN', np.array(y_test), knn_preds))
test_results.append(get_performance('Neural Network', np.array(y_test), nn_preds))

cv_results = np.reshape(np.array([round(f1_score(y_train, baseline),3), 0.563, 0.567, 0.623, 0.609, 0.521]),(6,1))

test_results = pd.DataFrame(test_results, columns = ['Model','Test F1 Score', 'Test Accuracy', 'Test Precision', 'Test Recall'])
test_results['CV F1 Score'] = cv_results
test_results['% Diff Test F1 and CV F1'] = round(((test_results['Test F1 Score']/test_results['CV F1 Score']) - 1),3)*100

test_results.head(6)


# In[248]:


a = plt.bar(test_results['Model'], test_results['Test F1 Score'], align='center', alpha=0.5)
b = plt.bar(test_results['Model'], test_results['CV F1 Score'], align='center', alpha=0.5)

plt.ylabel('F1 Score')
plt.title('Programming language usage')
 
plt.show()


# In[ ]:


epochs = list(range(1, 100))
print(epochs)
parameters = {'learning_rate':[0.1], 'n_estimators':epochs, 'max_depth':[5]}

g = GradientBoostingClassifier(verbose=1)
g_model = GridSearchCV(g, parameters, cv= 3, scoring = 'f1')
g_model.fit(x_train, y_train)
g_model_grid_search_results = pd.DataFrame(g_model.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(g_model_grid_search_results, 'n_estimators')


# In[227]:


epochs = list(range(1, 500))
print(epochs)
parameters = {'alpha':[0.01], 'hidden_layer_sizes':[128]
              ,'learning_rate_init':[0.00001],'activation':['relu']
            , 'solver':['lbfgs'], 'max_iter':epochs}

nn = MLPClassifier(verbose =1)
nn_model = GridSearchCV(nn, parameters, cv= 3, scoring = 'f1')
nn_model.fit(x_train, y_train)


# In[226]:


nn_model_grid_search_results = pd.DataFrame(nn_model.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(nn_model_grid_search_results, 'max_iter')


# In[255]:


iterations = [x*10 for x in range(1,99)]
parameters = {'n_estimators':iterations}
g = GradientBoostingClassifier(verbose =1, max_depth = 5, learning_rate = 0.1)
nn_model = GridSearchCV(g, parameters, cv= 3, scoring = 'f1')
nn_model.fit(x_train, y_train)

nn_model_grid_search_results = pd.DataFrame(nn_model.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(nn_model_grid_search_results, 'n_estimators')


# In[256]:


nn_model_grid_search_results = pd.DataFrame(nn_model.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(nn_model_grid_search_results, 'n_estimators')


# In[257]:


import datetime
a = datetime.datetime.now()
nn_clf.best_estimator_.fit(x_train, y_train)
b =datetime.datetime.now()
print(b-a)


# In[258]:


a = datetime.datetime.now()
gbm_clf.best_estimator_.fit(x_train, y_train)
b =datetime.datetime.now()
print(b-a)


# In[259]:


a = datetime.datetime.now()
dt_clf.best_estimator_.fit(x_train, y_train)
b =datetime.datetime.now()
print(b-a)


# In[260]:


a = datetime.datetime.now()
knn_clf.best_estimator_.fit(x_train, y_train)
b =datetime.datetime.now()
print(b-a)


# In[261]:


a = datetime.datetime.now()
svm_clf.best_estimator_.fit(x_train, y_train)
b =datetime.datetime.now()
print(b-a)

