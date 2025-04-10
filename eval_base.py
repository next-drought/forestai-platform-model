import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import scipy.optimize as opt
from sklearn import svm
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase

def eval(year, n_feature_elimination=10, SPCDs=[131,110,111,121]):

	path = 'plot_all/base/'+year
	if not os.path.exists(path):
	    os.makedirs(path)

	path = 'result_all/base/'+year
	if not os.path.exists(path):
	    os.makedirs(path)
	   
	y_data = pd.read_csv('dataset/presence_data.csv')
	x_data = pd.read_csv('dataset/feature_' + year + '.csv')
	x_data.drop(x_data.columns[0], axis=1,inplace=True)
	join_data = pd.concat([y_data, x_data], axis=1)

	# Outlier Elimination
	feature_names = join_data.columns[6:].tolist()
	join_data = join_data[(np.abs(stats.zscore(join_data[feature_names])) < 3).all(axis=1)]

	x_data = join_data.iloc[:,6:]
	y_data = join_data.iloc[:,:6]

	# Normalization
	mean = x_data.mean(axis=0)
	std = x_data.std(axis=0)
	x_data = (x_data - mean) / std

# 	vif_data = pd.DataFrame() 
# 	vif_data["feature"] = x_data.columns
# 	vif_data["VIF"] = [variance_inflation_factor(x_data.values, i) 
# 		           for i in range(len(x_data.columns))] 

# 	f = open('VIF_'+ year +'.txt','w')
# 	print(vif_data, file = f)
# 	for _ in range(n_feature_elimination):
# 	    largest_vif_id = vif_data['VIF'].idxmax()
# 	    x_data.drop(x_data.columns[largest_vif_id], axis=1,inplace=True)
# 	    vif_data = pd.DataFrame() 
# 	    vif_data["feature"] = x_data.columns
# 	    vif_data["VIF"] = [variance_inflation_factor(x_data.values, i) 
# 		               for i in range(len(x_data.columns))] 
# 	print('----------------------------------------\n', file = f)
# 	print(vif_data, file = f)
# 	f.close()
	join_data = pd.concat([y_data, x_data], axis=1)
	species_name = {131: "loblolly pine", 111: "slash pine", 110: "shortleaf pine", 121: "longleaf pine"}
	species_n = {111:14000, 110:24000, 121:10000}
	for SPCD in SPCDs:
		with open("dataset/background_"+str(SPCD), "r") as fp:
		    non_species = json.load(fp)
		    
		
		species_data = join_data[join_data['SPCD'] == SPCD]
		non_species_data = join_data[join_data['index'].isin(non_species)]
		non_species_data_a = non_species_data.assign(PRESENCE = 0)

		if SPCD == 131:
		    species_data_a = species_data.sample(n=20000, random_state = 42)
		    join_data_a = pd.concat([species_data_a, non_species_data_a], axis=0)
		else:
		    non_species_data_b = non_species_data_a.sample(n=species_n[SPCD], random_state = 42)
		    join_data_a = pd.concat([species_data, non_species_data_b], axis=0)


		x_data = join_data_a.iloc[:,6:]
		y_data = join_data_a.iloc[:,5]
		
		
		stats_label = ["Accuracy", "ROC AUC", "Kappa"]
		sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
		
		#svm
		ROC_AUC = []
		Accuracy = []
		Kappa = []
		clf = svm.SVC(kernel='poly', probability=True)
		for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
		    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
		    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
		    # print(cv_result)
		    # print(y_pred)
		    clf.fit(X_train, y_train)
		    y_pred = clf.predict(X_test)
		    # print("fold:"+str(i))
		    Accuracy.append(metrics.accuracy_score(y_test, y_pred))
		    ROC_AUC.append(metrics.roc_auc_score(y_test, y_pred))
		    Kappa.append(metrics.cohen_kappa_score(y_test, y_pred))
		    # y_pred = clf.predict_proba(X_test)
		svm_stats = [np.mean(Accuracy), np.mean(ROC_AUC), np.mean(Kappa)]
		# print("Accuracy:",np.mean(Accuracy))
		# print("ROC AUC:",np.mean(ROC_AUC))
		# print("Kappa:",np.mean(Kappa))
		
		#glm
		ROC_AUC = []
		Accuracy = []
		Kappa = []
		for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
		    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
		    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
		    X_glm_train = sm.add_constant(X_train)
		    glm = sm.GLM(y_train, X_glm_train, family=sm.families.Binomial())
		    result = glm.fit()
		    X_glm_test = sm.add_constant(X_test)
		    y_pred = result.predict(X_glm_test)
		    y_pred = [ 0 if x < 0.5 else 1 for x in y_pred]
		    Accuracy.append(metrics.accuracy_score(y_test, y_pred))
		    ROC_AUC.append(metrics.roc_auc_score(y_test, y_pred))
		    Kappa.append(metrics.cohen_kappa_score(y_test, y_pred))

		glm_stats = [np.mean(Accuracy), np.mean(ROC_AUC), np.mean(Kappa)]
		
		
		#gbm
		ROC_AUC = []
		Accuracy = []
		Kappa = []
		model = GradientBoostingClassifier()
		for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
		    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
		    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
		    model.fit(X_train, y_train)
		    y_pred = model.predict(X_test)
		    Accuracy.append(metrics.accuracy_score(y_test, y_pred))
		    ROC_AUC.append(metrics.roc_auc_score(y_test, y_pred))
		    Kappa.append(metrics.cohen_kappa_score(y_test, y_pred))

		gbm_stats = [np.mean(Accuracy), np.mean(ROC_AUC), np.mean(Kappa)]
		
		
		#rf
		ROC_AUC = []
		Accuracy = []
		Kappa = []

		rf = RandomForestClassifier()
		for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
		    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
		    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

		    rf.fit(X_train, y_train)
		    y_pred = rf.predict(X_test)
		    
		    Accuracy.append(metrics.accuracy_score(y_test, y_pred))
		    ROC_AUC.append(metrics.roc_auc_score(y_test, y_pred))
		    Kappa.append(metrics.cohen_kappa_score(y_test, y_pred))

		rf_stats = [np.mean(Accuracy), np.mean(ROC_AUC), np.mean(Kappa)]
		
		#nbm
		ROC_AUC = []
		Accuracy = []
		Kappa = []

		gnb = GaussianNB()

		for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
		    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
		    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
		    gnb.fit(X_train, y_train)
		    y_pred = gnb.predict(X_test)
		    Accuracy.append(metrics.accuracy_score(y_test, y_pred))
		    ROC_AUC.append(metrics.roc_auc_score(y_test, y_pred))
		    Kappa.append(metrics.cohen_kappa_score(y_test, y_pred))

		nbm_stats = [np.mean(Accuracy), np.mean(ROC_AUC), np.mean(Kappa)]
		
		#xgb
		ROC_AUC = []
		Accuracy = []
		Kappa = []
		xgb_classifier = xgb.XGBClassifier()
		for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
		    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
		    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
		    xgb_classifier.fit(X_train, y_train)
		    y_pred = xgb_classifier.predict(X_test)
		    Accuracy.append(metrics.accuracy_score(y_test, y_pred))
		    ROC_AUC.append(metrics.roc_auc_score(y_test, y_pred))
		    Kappa.append(metrics.cohen_kappa_score(y_test, y_pred))

		xgb_stats = [np.mean(Accuracy), np.mean(ROC_AUC), np.mean(Kappa)]
		
		
		#maxent
		ROC_AUC = []
		Accuracy = []
		Kappa = []
		logisticRegr = LogisticRegression(max_iter=1000)
		for i, (train_index, test_index) in enumerate(sss.split(x_data, y_data)):
		    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
		    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
		    logisticRegr.fit(X_train, y_train)
		    y_pred = logisticRegr.predict(X_test)
		    Accuracy.append(metrics.accuracy_score(y_test, y_pred))
		    ROC_AUC.append(metrics.roc_auc_score(y_test, y_pred))
		    Kappa.append(metrics.cohen_kappa_score(y_test, y_pred))

		maxent_stats = [np.mean(Accuracy), np.mean(ROC_AUC), np.mean(Kappa)]
		
		fig, ax = plt.subplots()
		ax.scatter(stats_label, svm_stats, label="SVM")
		ax.scatter(stats_label, glm_stats, label="GLM")
		ax.scatter(stats_label, gbm_stats, label="GBM")
		ax.scatter(stats_label, rf_stats, label="RF")
		ax.scatter(stats_label, nbm_stats, label="NBM")
		ax.scatter(stats_label, xgb_stats, label="XGB")
		ax.scatter(stats_label, maxent_stats, label="MAXENT")
		# ax.legend()
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.ylabel("Evaluation metric")
		plt.title("Bubble diagram for "+ species_name[SPCD] + " with "+year+" data")
		# plt.show()
		plt.savefig('plot_all/base/'+year+'/'+year+'_'+str(SPCD)+'_metrics.png')
		plt.close()
		name = ["SVM", "GLM","GBM","RF","NBM","XGB","MAXENT"]
		result = pd.DataFrame([svm_stats,glm_stats,gbm_stats,rf_stats,nbm_stats,xgb_stats,maxent_stats], index = name, columns = stats_label)
		# print(result)
		result.to_csv('result_all/base/'+year+'/'+year+'_'+str(SPCD)+'_metrics.csv')
		


