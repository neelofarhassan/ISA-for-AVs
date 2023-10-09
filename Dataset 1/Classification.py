#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
import scipy.stats as stats
from sklearn.model_selection import train_test_split

classification_report_dict = {}
precision = recall = f1_score = 0.0
# 2d lists for adding precision, recall and f1-score
random_random_forest_performance = [[] for _ in range(3)] 
random_decision_tree_performance = [[] for _ in range(3)]
random_k_neighbours_performance = [[] for _ in range(3)]
random_multilayer_percepteron_performance = [[] for _ in range(3)]
random_naive_bayes_performance = [[] for _ in range(3)]

ISA_random_forest_performance = [[] for _ in range(3)] 
ISA_decision_tree_performance = [[] for _ in range(3)]
ISA_k_neighbours_performance = [[] for _ in range(3)]
ISA_multilayer_percepteron_performance = [[] for _ in range(3)]
ISA_naive_bayes_performance = [[] for _ in range(3)]

iterations = 10
for _ in range(iterations):

  df_complete=pd.read_csv('complete-feature-set.csv')
  df_metadata = pd.read_csv('metadata.csv')


  #For random feature selection
  random_features_dataframe = df_complete.loc[:, df_complete.columns.str.contains('feature')].sample(n=10, axis='columns')
  random_feature_labels = list(random_features_dataframe.columns.values)
  print('feature labels random: ' , random_feature_labels)
  random_algo_dataframe = df_complete.loc[:, df_complete.columns.str.contains('algo')] 
  target_names = random_algo_dataframe.columns
  random_train_features, random_test_features, random_train_labels, random_test_labels = train_test_split(random_features_dataframe, random_algo_dataframe, test_size = 0.25, random_state = 42)

  # random_train_features=np.array(random_features_dataframe)
  # random_train_labels=np.array(random_algo_dataframe)

  # For ISA feature selection
  ISA_features = ['feature_ego_brake',	'feature_ego_speed', 'feature_scenarioTrafficLightDemand', 
                  'feature_totalNPCs', 'feature_isPedestrianScenario', 'feature_totalRoadUsers', 
                  'feature_obstaclesMinimumDistance',	'feature_speedObstacleWithMinimumDistance', 
                  'feature_volumeObstacleWithMinimumDistance', 'feature_obstaclesMaximumSpeed']
  ISA_features_dataframe = df_metadata[ISA_features]
  ISA_feature_labels = list(ISA_features_dataframe.columns.values)
  print('feature labels ISA', ISA_feature_labels)
  ISA_algo_dataframe = df_metadata.loc[:, df_metadata.columns.str.contains('algo')] 
  ISA_train_features, ISA_test_features, ISA_train_labels, ISA_test_labels = train_test_split(ISA_features_dataframe, ISA_algo_dataframe, test_size = 0.25, random_state = 42)
  # ISA_train_features = np.array(ISA_features_dataframe)  
  # ISA_train_labels = np.array(ISA_algo_dataframe)
  

  from sklearn.preprocessing import MinMaxScaler
  scaling = MinMaxScaler(feature_range=(-1,1)).fit(random_train_features)
  random_train_features = scaling.transform(random_train_features)

  scaling = MinMaxScaler(feature_range=(-1,1)).fit(ISA_train_features)
  ISA_train_features = scaling.transform(ISA_train_features)


  # "***********************Random Forest***********************"
  random_rf = RandomForestClassifier(n_estimators=100, random_state=1)
  random_rf.fit(random_train_features, np.ravel(random_train_labels))

  ISA_rf = RandomForestClassifier(n_estimators=100, random_state=1)
  ISA_rf.fit(ISA_train_features, np.ravel(ISA_train_labels))

  # "***********************Decision Tree***********************"
  random_clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
  random_clf_en.fit(random_train_features, np.ravel(random_train_labels))

  ISA_clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
  ISA_clf_en.fit(ISA_train_features, np.ravel(ISA_train_labels))


  # "***********************K Nearest Neighbour***********************
  random_knn = KNeighborsClassifier(n_neighbors=5)
  random_knn.fit(random_train_features, np.ravel(random_train_labels))

  ISA_knn = KNeighborsClassifier(n_neighbors=5)
  ISA_knn.fit(ISA_train_features, np.ravel(ISA_train_labels))

  # ***********************Multilayer Perceptron***********************"
  from sklearn.neural_network import MLPClassifier
  random_MLP = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
  random_MLP.fit(random_train_features, np.ravel(random_train_labels))

  ISA_MLP = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
  ISA_MLP.fit(ISA_train_features, np.ravel(ISA_train_labels))

  # "***********************Naive Bayes***********************
  random_GNB = GaussianNB()
  random_GNB.fit(random_train_features, np.ravel(random_train_labels))

  ISA_GNB = GaussianNB()
  ISA_GNB.fit(ISA_train_features, np.ravel(ISA_train_labels))


 
  from sklearn.preprocessing import MinMaxScaler
  scaling = MinMaxScaler(feature_range=(-1,1)).fit(random_test_features)
  random_test_features = scaling.transform(random_test_features)


  scaling = MinMaxScaler(feature_range=(-1,1)).fit(ISA_test_features)
  ISA_test_features = scaling.transform(ISA_test_features)

  print('Testing Features Shape:', random_test_features.shape)
  print('Testing Labels Shape:', random_test_labels.shape)


  # print("***********************Random Forest***********************")
  # print("Random")

  predictions = random_rf.predict(random_test_features)
  classification_report_dict = classification_report(random_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  random_random_forest_performance[0].append(precision)
  random_random_forest_performance[1].append(recall)
  random_random_forest_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()

  # print("ISA")
  predictions = ISA_rf.predict(ISA_test_features)
  classification_report_dict = classification_report(ISA_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  ISA_random_forest_performance[0].append(precision)
  ISA_random_forest_performance[1].append(recall)
  ISA_random_forest_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()

  # print("***********************Decision Tree***********************")
  # print("random")
  predictions = random_clf_en.predict(random_test_features)
  classification_report_dict = classification_report(random_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  random_decision_tree_performance[0].append(precision)
  random_decision_tree_performance[1].append(recall)
  random_decision_tree_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()

  # print("ISA ")
  predictions = ISA_clf_en.predict(ISA_test_features)
  classification_report_dict = classification_report(ISA_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  ISA_decision_tree_performance[0].append(precision)
  ISA_decision_tree_performance[1].append(recall)
  ISA_decision_tree_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()

  # print("***********************K Nearest Neighbour***********************")

  # print("random")
  predictions = random_knn.predict(random_test_features)
  classification_report_dict = classification_report(random_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  random_k_neighbours_performance[0].append(precision)
  random_k_neighbours_performance[1].append(recall)
  random_k_neighbours_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()


  # print("ISA")
  predictions = ISA_knn.predict(ISA_test_features)
  classification_report_dict = classification_report(ISA_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  ISA_k_neighbours_performance[0].append(precision)
  ISA_k_neighbours_performance[1].append(recall)
  ISA_k_neighbours_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()


  # print("***********************Multilayer Perceptron***********************")  
  # print("random")
  predictions = random_MLP.predict(random_test_features)
  classification_report_dict = classification_report(random_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  random_multilayer_percepteron_performance[0].append(precision)
  random_multilayer_percepteron_performance[1].append(recall)
  random_multilayer_percepteron_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()

  # print("ISA")
  predictions = ISA_MLP.predict(ISA_test_features)
  classification_report_dict = classification_report(ISA_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  ISA_multilayer_percepteron_performance[0].append(precision)
  ISA_multilayer_percepteron_performance[1].append(recall)
  ISA_multilayer_percepteron_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()


  # print("***********************Naive Bayes***********************")

  # print("random")
  predictions = random_GNB.predict(random_test_features)
  classification_report_dict = classification_report(random_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  random_naive_bayes_performance[0].append(precision)
  random_naive_bayes_performance[1].append(recall)
  random_naive_bayes_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()

  # print("ISA")
  predictions = ISA_GNB.predict(ISA_test_features)
  classification_report_dict = classification_report(ISA_test_labels, predictions, output_dict=True)
  macro_average = classification_report_dict.get('macro avg')
  precision = macro_average.get('precision')
  recall = macro_average.get('recall')
  f1_score = macro_average.get('f1-score')
  ISA_naive_bayes_performance[0].append(precision)
  ISA_naive_bayes_performance[1].append(recall)
  ISA_naive_bayes_performance[2].append(f1_score)
  macro_average.clear()
  classification_report_dict.clear()

  
print("random_forest_performance: ", random_random_forest_performance)
print("decision_tree_performance: ", random_decision_tree_performance)
print("naive_bayes_performance: ", random_naive_bayes_performance)
print("k_neighbours_performance: ", random_k_neighbours_performance)
print("multilayer_percepteron_performance: ", random_multilayer_percepteron_performance)

print("*****************Average Performance Random***********************")
print('Algorithm\t\t\t precision\t\t\t\trecall\t\t\t\tf1-score')
print('RF\t\t\t',  round(sum(random_random_forest_performance[0])/len(random_random_forest_performance[0]), 3), '\t\t\t\t', round(sum(random_random_forest_performance[1])/len(random_random_forest_performance[1]), 3), '\t\t\t\t', round(sum(random_random_forest_performance[2])/len(random_random_forest_performance[2]), 3))
print('DT\t\t\t',  round(sum(random_decision_tree_performance[0])/len(random_decision_tree_performance[0]), 3), '\t\t\t\t', round(sum(random_decision_tree_performance[1])/len(random_decision_tree_performance[1]), 3), '\t\t\t\t', round(sum(random_decision_tree_performance[2])/len(random_decision_tree_performance[2]), 3))
print('KNN\t\t\t',  round(sum(random_k_neighbours_performance[0])/len(random_k_neighbours_performance[0]), 3), '\t\t\t\t', round(sum(random_k_neighbours_performance[1])/len(random_k_neighbours_performance[1]), 3), '\t\t\t\t', round(sum(random_k_neighbours_performance[2])/len(random_k_neighbours_performance[2]), 3))
print('MLP\t\t\t',  round(sum(random_multilayer_percepteron_performance[0])/len(random_multilayer_percepteron_performance[0]), 3), '\t\t\t\t', round(sum(random_multilayer_percepteron_performance[1])/len(random_multilayer_percepteron_performance[1]), 3), '\t\t\t\t', round(sum(random_multilayer_percepteron_performance[2])/len(random_multilayer_percepteron_performance[2]), 3))
print('NV\t\t\t',  round(sum(random_naive_bayes_performance[0])/len(random_naive_bayes_performance[0]), 3), '\t\t\t\t', round(sum(random_naive_bayes_performance[1])/len(random_naive_bayes_performance[1]), 3), '\t\t\t\t', round(sum(random_naive_bayes_performance[2])/len(random_naive_bayes_performance[2]), 3))


print("*****************Average Performance ISA***********************")
print('Algorithm\t\t\t precision\t\t\t\trecall\t\t\t\tf1-score')
print('RF\t\t\t',  round(sum(ISA_random_forest_performance[0])/len(ISA_random_forest_performance[0]), 3), '\t\t\t\t', round(sum(ISA_random_forest_performance[1])/len(ISA_random_forest_performance[1]), 3), '\t\t\t\t', round(sum(ISA_random_forest_performance[2])/len(ISA_random_forest_performance[2]), 3))
print('DT\t\t\t',  round(sum(ISA_decision_tree_performance[0])/len(ISA_decision_tree_performance[0]), 3), '\t\t\t\t', round(sum(ISA_decision_tree_performance[1])/len(ISA_decision_tree_performance[1]), 3), '\t\t\t\t', round(sum(ISA_decision_tree_performance[2])/len(ISA_decision_tree_performance[2]), 3))
print('KNN\t\t\t',  round(sum(ISA_k_neighbours_performance[0])/len(ISA_k_neighbours_performance[0]), 3), '\t\t\t\t', round(sum(ISA_k_neighbours_performance[1])/len(ISA_k_neighbours_performance[1]), 3), '\t\t\t\t', round(sum(ISA_k_neighbours_performance[2])/len(ISA_k_neighbours_performance[2]), 3))
print('MLP\t\t\t',  round(sum(ISA_multilayer_percepteron_performance[0])/len(ISA_multilayer_percepteron_performance[0]), 3), '\t\t\t\t', round(sum(ISA_multilayer_percepteron_performance[1])/len(ISA_multilayer_percepteron_performance[1]), 3), '\t\t\t\t', round(sum(ISA_multilayer_percepteron_performance[2])/len(ISA_multilayer_percepteron_performance[2]), 3))
print('NV\t\t\t',  round(sum(ISA_naive_bayes_performance[0])/len(ISA_naive_bayes_performance[0]), 3), '\t\t\t\t', round(sum(ISA_naive_bayes_performance[1])/len(ISA_naive_bayes_performance[1]), 3), '\t\t\t\t', round(sum(ISA_naive_bayes_performance[2])/len(ISA_naive_bayes_performance[2]), 3))


df_random = pd.DataFrame()
df_random['id'] = [x for x in range(iterations)]
df_random['RF_precision'] = random_random_forest_performance[0]
df_random['RF_recall'] = random_random_forest_performance[1]
df_random['RF_f1'] = random_random_forest_performance[2]

df_random['DT_precision'] = random_decision_tree_performance[0]
df_random['DT_recall'] = random_decision_tree_performance[1]
df_random['DT_f1'] = random_decision_tree_performance[1]

df_random['NB_precision'] = random_naive_bayes_performance[0]
df_random['NB_recall'] = random_naive_bayes_performance[1]
df_random['NB_f1'] = random_naive_bayes_performance[2]

df_random['MLP_precision'] = random_multilayer_percepteron_performance[0]
df_random['MLP_recall'] = random_multilayer_percepteron_performance[1]
df_random['MLP_f1'] = random_multilayer_percepteron_performance[2]

df_random['KNN_precision'] = random_k_neighbours_performance[0]
df_random['KNN_recall'] = random_k_neighbours_performance[1]
df_random['KNN_f1'] = random_k_neighbours_performance[2]
print(df_random)

df_ISA = pd.DataFrame()
df_ISA['id'] = [x for x in range(iterations)]
df_ISA['RF_precision'] = ISA_random_forest_performance[0]
df_ISA['RF_recall'] = ISA_random_forest_performance[1]
df_ISA['RF_f1'] = ISA_random_forest_performance[2]

df_ISA['DT_precision'] = ISA_decision_tree_performance[0]
df_ISA['DT_recall'] = ISA_decision_tree_performance[1]
df_ISA['DT_f1'] = ISA_decision_tree_performance[1]

df_ISA['NB_precision'] = ISA_naive_bayes_performance[0]
df_ISA['NB_recall'] = ISA_naive_bayes_performance[1]
df_ISA['NB_f1'] = ISA_naive_bayes_performance[2]

df_ISA['MLP_precision'] = ISA_multilayer_percepteron_performance[0]
df_ISA['MLP_recall'] = ISA_multilayer_percepteron_performance[1]
df_ISA['MLP_f1'] = ISA_multilayer_percepteron_performance[2]

df_ISA['KNN_precision'] = ISA_k_neighbours_performance[0]
df_ISA['KNN_recall'] = ISA_k_neighbours_performance[1]
df_ISA['KNN_f1'] = ISA_k_neighbours_performance[2]
print(df_ISA)

# Performing wilxoson signed rank test
# Column definition
df_performance_comparison = pd.DataFrame(columns=['algo', 'ISA', 'Random', 'p-value'])

# Random Forest
ISA_average = round(sum(ISA_random_forest_performance[0])/len(ISA_random_forest_performance[0]), 3)
random_average = round(sum(random_random_forest_performance[0])/len(random_random_forest_performance[0]), 3)
pvalue_average = stats.wilcoxon(ISA_random_forest_performance[0], random_random_forest_performance[0])
df_performance_comparison.loc[len(df_performance_comparison)] = ['precision_RF', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_random_forest_performance[1])/len(ISA_random_forest_performance[1]), 3)
random_average = round(sum(random_random_forest_performance[1])/len(random_random_forest_performance[1]), 3)
pvalue_average = stats.wilcoxon(ISA_random_forest_performance[1], random_random_forest_performance[1])
df_performance_comparison.loc[len(df_performance_comparison)] = ['recall_RF', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_random_forest_performance[2])/len(ISA_random_forest_performance[2]), 3)
random_average = round(sum(random_random_forest_performance[2])/len(random_random_forest_performance[2]), 3)
pvalue_average = stats.wilcoxon(ISA_random_forest_performance[2], random_random_forest_performance[2])
df_performance_comparison.loc[len(df_performance_comparison)] = ['f1_RF', ISA_average, random_average, round(pvalue_average[1],3)]

# Decision Tree
ISA_average = round(sum(ISA_decision_tree_performance[0])/len(ISA_decision_tree_performance[0]), 3)
random_average = round(sum(random_decision_tree_performance[0])/len(random_decision_tree_performance[0]), 3)
pvalue_average = stats.wilcoxon(ISA_decision_tree_performance[0], random_decision_tree_performance[0])
df_performance_comparison.loc[len(df_performance_comparison)] = ['precision_DT', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_decision_tree_performance[1])/len(ISA_decision_tree_performance[1]), 3)
random_average = round(sum(random_decision_tree_performance[1])/len(random_decision_tree_performance[1]), 3)
pvalue_average = stats.wilcoxon(ISA_decision_tree_performance[1], random_decision_tree_performance[1])
df_performance_comparison.loc[len(df_performance_comparison)] = ['recall_DT', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_decision_tree_performance[2])/len(ISA_decision_tree_performance[2]), 3)
random_average = round(sum(random_decision_tree_performance[2])/len(random_decision_tree_performance[2]), 3)
pvalue_average = stats.wilcoxon(ISA_decision_tree_performance[2], random_decision_tree_performance[2])
df_performance_comparison.loc[len(df_performance_comparison)] = ['f1_DT', ISA_average, random_average, round(pvalue_average[1],3)]

# KNN
ISA_average = round(sum(ISA_k_neighbours_performance[0])/len(ISA_k_neighbours_performance[0]), 3)
random_average = round(sum(random_k_neighbours_performance[0])/len(random_k_neighbours_performance[0]), 3)
pvalue_average = stats.wilcoxon(ISA_k_neighbours_performance[0], random_k_neighbours_performance[0])
df_performance_comparison.loc[len(df_performance_comparison)] = ['precision_KNN', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_k_neighbours_performance[1])/len(ISA_k_neighbours_performance[1]), 3)
random_average = round(sum(random_k_neighbours_performance[1])/len(random_k_neighbours_performance[1]), 3)
pvalue_average = stats.wilcoxon(ISA_k_neighbours_performance[1], random_k_neighbours_performance[1])
df_performance_comparison.loc[len(df_performance_comparison)] = ['recall_KNN', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_k_neighbours_performance[2])/len(ISA_k_neighbours_performance[2]), 3)
random_average = round(sum(random_k_neighbours_performance[2])/len(random_k_neighbours_performance[2]), 3)
pvalue_average = stats.wilcoxon(ISA_k_neighbours_performance[2], random_k_neighbours_performance[2])
df_performance_comparison.loc[len(df_performance_comparison)] = ['f1_KNN', ISA_average, random_average, round(pvalue_average[1],3)]

# multilayer_percepteron
ISA_average = round(sum(ISA_multilayer_percepteron_performance[0])/len(ISA_multilayer_percepteron_performance[0]), 3)
random_average = round(sum(random_multilayer_percepteron_performance[0])/len(random_multilayer_percepteron_performance[0]), 3)
pvalue_average = stats.wilcoxon(ISA_multilayer_percepteron_performance[0], random_multilayer_percepteron_performance[0])
df_performance_comparison.loc[len(df_performance_comparison)] = ['precision_MLP', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_multilayer_percepteron_performance[1])/len(ISA_multilayer_percepteron_performance[1]), 3)
random_average = round(sum(random_multilayer_percepteron_performance[1])/len(random_multilayer_percepteron_performance[1]), 3)
pvalue_average = stats.wilcoxon(ISA_multilayer_percepteron_performance[1], random_multilayer_percepteron_performance[1])
df_performance_comparison.loc[len(df_performance_comparison)] = ['recall_MLP', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_multilayer_percepteron_performance[2])/len(ISA_multilayer_percepteron_performance[2]), 3)
random_average = round(sum(random_multilayer_percepteron_performance[2])/len(random_multilayer_percepteron_performance[2]), 3)
pvalue_average = stats.wilcoxon(ISA_multilayer_percepteron_performance[2], random_multilayer_percepteron_performance[2])
df_performance_comparison.loc[len(df_performance_comparison)] = ['f1_MLP', ISA_average, random_average, round(pvalue_average[1],3)]

# Naive Bayes
ISA_average = round(sum(ISA_naive_bayes_performance[0])/len(ISA_naive_bayes_performance[0]), 3)
random_average = round(sum(random_naive_bayes_performance[0])/len(random_naive_bayes_performance[0]), 3)
pvalue_average = stats.wilcoxon(ISA_naive_bayes_performance[0], random_naive_bayes_performance[0])
df_performance_comparison.loc[len(df_performance_comparison)] = ['precision_NB', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_naive_bayes_performance[1])/len(ISA_naive_bayes_performance[1]), 3)
random_average = round(sum(random_naive_bayes_performance[1])/len(random_naive_bayes_performance[1]), 3)
pvalue_average = stats.wilcoxon(ISA_naive_bayes_performance[1], random_naive_bayes_performance[1])
df_performance_comparison.loc[len(df_performance_comparison)] = ['recall_NB', ISA_average, random_average, round(pvalue_average[1],3)]

ISA_average = round(sum(ISA_naive_bayes_performance[2])/len(ISA_naive_bayes_performance[2]), 3)
random_average = round(sum(random_naive_bayes_performance[2])/len(random_naive_bayes_performance[2]), 3)
pvalue_average = stats.wilcoxon(ISA_naive_bayes_performance[2], random_multilayer_percepteron_performance[2])
df_performance_comparison.loc[len(df_performance_comparison)] = ['f1_NB', ISA_average, random_average, round(pvalue_average[1],3)]

print(df_performance_comparison)
df_performance_comparison.to_csv('path-to-output-directory/performance.csv')


