import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
import time
import warnings
warnings.filterwarnings('ignore')

raw_data = pd.read_csv("../data/continuous_factory_process.csv", index_col="time_stamp")
print(raw_data.info())
# print(raw_data.head())

#Drop setpoint data
print("*************************Remove set point from raw data*****************************")
set_point = [42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114]
set_point_name = list(raw_data.columns[set_point])
set_data = raw_data.drop(columns = set_point_name, axis=1)
print(set_data.info())

# Drop by missing rate
print("***************************Check 0 more than 30% in columns**************************")
Araw_data = set_data.loc[:, set_data.eq(0).mean() < 0.3]
# print(new_data.info())

Input_A = Araw_data.values[:,0:41]
Input_AN = Araw_data.columns[0:41]
Input_AA = pd.DataFrame(data=Input_A, columns=Input_AN)
Input_AAN = pd.DataFrame(data=Input_A) #Non name
Output_A = Araw_data.values[:,41:50]
Output_AN = Araw_data.columns[41:50]
Output_AA = pd.DataFrame(data=Output_A, columns=Output_AN)
print("Input for model A")
print(Input_AA.info())
print("Output for model A")
print(Output_AA.info())



#Feature selection and prediction for model A
print("********************************Prediction model A (Stage 1)*******************************")
for n in range (len(Output_AN)):
    Yi = Output_AA.values [:,n];
    # Backward Elimination #for feature selection
    cols = list(Input_AAN.columns)
    pmax = 1
    while (len(cols) > 0):
       p = []
       X_1 = Input_AAN[cols]
       X_1 = sm.add_constant(X_1)
       model = sm.OLS(Yi, X_1).fit()
       p = pd.Series(model.pvalues.values[1:], index=cols)
       pmax = max(p)
       feature_with_p_max = p.idxmax()
       if (pmax > 0.05):
          cols.remove(feature_with_p_max)
       else:
          break
    selected_features = cols
    #print(Output_AN[n], "(Selected feature) =", list(Input_AA.columns[cols])) #list of selected feature model A
    print(Output_AN[n],"     Total feature","(",len(Input_AN),")","Selected feature",len(selected_features))
    x_selected = Input_AA.values[:, selected_features]
    x_train, x_test, y_train, y_test = train_test_split(x_selected, Yi, test_size=0.3)

    # SVM-poly
    svr_poly = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2, kernel='poly', degree=3))
    svr_poly = svr_poly.fit(x_train, y_train)
    Svr_poly = abs(svr_poly.score(x_test, y_test))
    Svr_poly = "{:.2f}".format(Svr_poly * 100)

    # Decision tree
    clf = tree.DecisionTreeRegressor(max_features='auto')
    clf = clf.fit(x_train, y_train)
    CRF = abs(clf.score(x_test, y_test))
    CRF = "{:.2f}".format(CRF * 100)

    # KNN
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2,
                                metric='minkowski', metric_params=None, n_jobs=None)
    neigh = neigh.fit(x_train, y_train)
    KNN = abs(neigh.score(x_test, y_test))
    KNN = "{:.2f}".format(KNN * 100)

    # Accuracy
    print("             %Prediction accuracy,", "Tree ", CRF, " SVM-Poly ", Svr_poly, "KNN ", KNN)