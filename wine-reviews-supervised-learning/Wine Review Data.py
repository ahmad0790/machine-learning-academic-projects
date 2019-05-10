
# coding: utf-8

# In[61]:


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


# In[2]:


wine = pd.read_csv('wine_reviews.csv')
wine = wine.sample(n=5000)


# In[14]:


wine[['description','country','variety']].head()


# In[5]:


wine['points'].hist()


# In[6]:


wine['points'][wine['points']<92] = 0
wine['points'][wine['points']>=92] = 1


# In[7]:


corpus = wine['description']
stemmer = PorterStemmer()
processed_reviews = []
for review in corpus:
    review_tokenized = word_tokenize(review)
    new_review = []
    for word in review_tokenized:
        new_review.append(stemmer.stem(word))
    new_sentence = ' '.join(str(e) for e in new_review)
    processed_reviews.append(new_sentence)

print(processed_reviews[0:10])


# In[8]:


vectorizer = CountVectorizer(stop_words = 'english', max_features =150, binary = True)
review_words = vectorizer.fit_transform(processed_reviews)
review_words = review_words.toarray()
review_words.shape


# In[26]:


threshold = 50 
country_counts = wine.country.value_counts()
repl = country_counts[country_counts <= threshold].index
countries = pd.get_dummies(wine.country.replace(repl, 'uncommon'))
countries.head()


# In[35]:


threshold = 22
variety_counts = wine.variety.value_counts()
repl = variety_counts[variety_counts <= threshold].index
varieties = pd.get_dummies(wine.variety.replace(repl, 'uncommon'))
varieties.head()


# In[57]:


wine_data = np.hstack((countries.values, varieties.values, review_words))
wine_data.shape


# In[50]:


wine_quality = wine['points']
wine['points'].value_counts()


# In[67]:


x_train, x_test, y_train, y_test = train_test_split(wine_data, np.array(wine_quality), test_size=0.3, random_state=0)


# In[68]:


print("Train Rows:" + str(x_train.shape))
print("Train Labels:" + str(y_train.shape))
print("Test Rows:" + str(x_test.shape))
print("Test Labels" + str(y_test.shape))


# In[84]:


#baseline
baseline = random.choices(population=[0,1],weights=[0.80, 0.20],k=x_train.shape[0])
print("Baseline Accuracy:" + str(sklearn.metrics.accuracy_score(y_train, baseline)))
print("Baseline Precision:" + str(sklearn.metrics.precision_score(y_train, baseline)))
print("Baseline Recall:" + str(sklearn.metrics.recall_score(y_train, baseline)))
print("Baseline F1 Score:" + str(sklearn.metrics.f1_score(y_train, baseline)))


# In[176]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


# In[86]:


def plot_model_complexity(grid_search_results, param_name):
    param = 'param_'+str(param_name)
    means_train = grid_search_results.groupby([param])['mean_train_score'].mean().reset_index()
    means_cv = grid_search_results.groupby([param])['mean_test_score'].mean().reset_index()
    means_train.sort_values(by=[param], inplace=True)
    means_cv.sort_values(by=[param], inplace=True)    
    param_values = list(means_train[param])
    
    plt.figure(figsize=(8, 8))
    plt.plot(param_values, means_train['mean_train_score'], 'r')
    plt.plot(param_values, means_cv['mean_test_score'], 'b')
    plt.title(param_name + " vs F1 Score")
    plt.xlabel(param_name)
    plt.ylabel('F1 Score')
    plt.ylim(means_cv['mean_test_score'].min()-0.1)
    plt.legend()
    plt.show()


# In[142]:


#Decisison Tree Hyperparameter Impact
dt_parameters = {'min_samples_leaf':[1,5, 10, 20, 25,50]}
dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, dt_parameters,cv=5, scoring = 'f1')
clf.fit(x_train, y_train)
dt_grid_search_results = pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(dt_grid_search_results, 'min_samples_leaf')


# In[143]:


dt_parameters = {'max_depth':[1, 5, 10, 20, 25,50]}
dt = DecisionTreeClassifier()
dt_clf = GridSearchCV(dt, dt_parameters,cv=5, scoring = 'f1')
dt_clf.fit(x_train, y_train)
dt_grid_search_results = pd.DataFrame(dt_clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(dt_grid_search_results, 'max_depth')


# In[177]:


#Decisison Tree Grid Search
dt_parameters = {'max_depth':[5,10,15,25,50], 'min_samples_leaf':[1, 5, 10, 20, 25, 50]}
dt = DecisionTreeClassifier()
dt_clf = GridSearchCV(dt, dt_parameters,cv=cv, scoring = 'f1')
dt_clf.fit(x_train, y_train)


# In[178]:


dt_grid_search_results = pd.DataFrame(dt_clf.cv_results_).sort_values(by='rank_test_score')


# In[180]:


# View the best parameters for the model found using grid search
print('Best score for data1:', dt_clf.best_score_) 
print('Best Max Depth:',dt_clf.best_estimator_.max_depth) 
print('Best Shrinkage:',dt_clf.best_estimator_.min_samples_leaf)


# In[201]:


#SVM C Search
parameters = {'C':[0.001,0.01,0.1,1,10]}
svm = SVC(verbose =0, kernel='linear')
svm_clf = GridSearchCV(svm, parameters, cv= cv, scoring = 'f1')
svm_clf.fit(x_train, y_train)
svm_grid_search_results = pd.DataFrame(svm_clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(svm_grid_search_results, 'C')


# In[202]:


#SVM C Search
parameters = {'C':[0.001,0.01,0.1,1,10]}
svm = SVC(verbose =0, kernel='rbf')
svm_clf = GridSearchCV(svm, parameters, cv= cv, scoring = 'f1')
svm_clf.fit(x_train, y_train)
svm_grid_search_results = pd.DataFrame(svm_clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(svm_grid_search_results, 'C')


# In[199]:


#SVM gamma Search
parameters = {'gamma':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1]}
svm = SVC(verbose =0, kernel='rbf')
svm_clf = GridSearchCV(svm, parameters, cv= cv, scoring = 'f1')
svm_clf.fit(x_train, y_train)
svm_grid_search_results = pd.DataFrame(svm_clf.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(svm_grid_search_results, 'gamma')


# In[169]:


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


# In[168]:


#GBM Grid Search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
parameters = {'learning_rate':[0.001, 0.01, 0.1, 0.3, 0.5], 'n_estimators':[10,100,250,500,1000], 'max_depth':[1,3,5,7,10]}
gbm = GradientBoostingClassifier(verbose =1)
gbm_clf = GridSearchCV(gbm, parameters, cv= cv, scoring = 'f1')
gbm_clf.fit(x_train, y_train)


# In[181]:


gbm_grid_search_results = pd.DataFrame(gbm_clf.cv_results_).sort_values(by='rank_test_score')


# In[183]:


# View the best parameters for the model found using grid search
print('Best score for data1:', gbm_clf.best_score_) 
print('Best Estimators:',gbm_clf.best_estimator_.n_estimators) 
print('Best Shrinkage:',gbm_clf.best_estimator_.learning_rate)
print('Best Shrinkage:',gbm_clf.best_estimator_.max_depth)


# In[160]:


plot_model_complexity(gbm_grid_search_results, 'learning_rate')


# In[161]:


plot_model_complexity(gbm_grid_search_results, 'n_estimators')


# In[184]:


plot_model_complexity(gbm_grid_search_results, 'max_depth')


# In[ ]:


#GBM Grid Search
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
parameters = {'n_estimators':[10,100,250,500,1000]}
gbm_v2 = GradientBoostingClassifier(verbose =1)
gbm_epochs = GridSearchCV(gbm_v2, parameters, cv= cv, scoring = 'f1')
gbm_epochs.fit(x_train, y_train)


# In[162]:


#KNN Grid Search
parameters = {'n_neighbors':[1,3,5,7,10,25]}
knn = KNeighborsClassifier()
knn_clf = GridSearchCV(knn, parameters, cv= cv, scoring = 'f1')
knn_clf.fit(x_train, y_train)


# In[163]:


knn_grid_search_results = pd.DataFrame(knn_clf.cv_results_).sort_values(by='rank_test_score')


# In[164]:


# View the best parameters for the model found using grid search
print('Best score for data1:', knn_clf.best_score_) 
print('Best K:',knn_clf.best_estimator_.n_neighbors) 


# In[165]:


plot_model_complexity(knn_grid_search_results, 'n_neighbors')


# In[167]:


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
print('Best Learning Rate:',nn_clf.best_estimator_.learning_rate_init)
print('Best HL:',nn_clf.best_estimator_.hidden_layer_sizes)
print('Best Activation:',nn_clf.best_estimator_.activation)
print('Best Optimizer:',nn_clf.best_estimator_.solver)


# In[170]:


print('Best Learning Rate:',nn_clf.best_estimator_.learning_rate_init)


# In[171]:


nn_grid_search_results = pd.DataFrame(nn_clf.cv_results_).sort_values(by='rank_test_score')


# In[ ]:


# View the best parameters for the model found using grid search
print('Best score for data1:', nn_clf.best_score_) 
print('Best Estimators:',nn_clf.best_estimator_.alpha) 
print('Best Shrinkage:',nn_clf.best_estimator_.epsilon)
print('Best Shrinkage:',nn_clf.best_estimator_.hidden_layer_size)


# In[172]:


plot_model_complexity(nn_grid_search_results, 'alpha')


# In[173]:


plot_model_complexity(nn_grid_search_results, 'hidden_layer_sizes')


# In[174]:


plot_model_complexity(nn_grid_search_results, 'learning_rate_init')


# In[113]:


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


# In[198]:


# SVM Learning Curve
#cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cv = StratifiedKFold(n_splits=5, random_state=0)
plot_learning_curve(svm_clf.best_estimator_, "Learning Curves for SVM Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=4)
plt.show()


# In[188]:


#DT Learning Curve
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
plot_learning_curve(dt_clf.best_estimator_, "Learning Curves for  DT Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=4)
plt.show()


# In[187]:


#GBM Learning Curve
plot_learning_curve(gbm_clf.best_estimator_, "Learning Curves for GBM Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=4)
plt.show()


# In[186]:


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
plot_learning_curve(MLPClassifier(alpha=0.01,learning_rate_init=0.001,hidden_layer_sizes=32
                                  ,activation='relu',solver='adam',verbose=False)
                                  ,"Learning Curve for NN Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=1)
plt.show()


# In[121]:


#KNN Learning Curve
plot_learning_curve(knn_clf.best_estimator_, "Learning Curves for KNN Model"
                    , x_train, y_train, (0.1, 1.1), cv=cv, n_jobs=4)
plt.show()


# In[197]:


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

cv_results = np.reshape(np.array([round(f1_score(y_train, baseline),3), 0.362, 0.422, 0.436, 0.277, 0.477]),(6,1))

test_results = pd.DataFrame(test_results, columns = ['Model','Test F1 Score', 'Test Accuracy', 'Test Precision', 'Test Recall'])
test_results['CV F1 Score'] = cv_results
test_results['% Diff Test F1 and CV F1'] = round(((test_results['Test F1 Score']/test_results['CV F1 Score']) - 1),3)*100

test_results.head(6)


# In[209]:


epochs = [x*5 for x in range(1,20)]
print(epochs)
parameters = {'alpha':[0.01], 'hidden_layer_sizes':[32]
              ,'learning_rate_init':[0.001],'activation':['relu']
            , 'solver':['adam'], 'max_iter':epochs}

nn = MLPClassifier(verbose =1, early_stopping=False, tol = 0.0000001)
nn_model = GridSearchCV(nn, parameters, cv= 3, scoring = 'f1')
nn_model.fit(x_train, y_train)


# In[210]:


nn_model_grid_search_results = pd.DataFrame(nn_model.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(nn_model_grid_search_results, 'max_iter')


# In[211]:


iterations = [x*10 for x in range(1,99)]
parameters = {'n_estimators':iterations}
g = GradientBoostingClassifier(verbose =1, max_depth = 3, learning_rate = 0.3)
nn_model = GridSearchCV(g, parameters, cv= 3, scoring = 'f1')
nn_model.fit(x_train, y_train)


# In[212]:


nn_model_grid_search_results = pd.DataFrame(nn_model.cv_results_).sort_values(by='rank_test_score')
plot_model_complexity(nn_model_grid_search_results, 'n_estimators')

