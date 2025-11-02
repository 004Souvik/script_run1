import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cross_decomposition import PLSRegression

def metrics(obs_tr, pred_tr, obs_te, pred_te, method):
    r2 = round(r2_score(obs_tr, pred_tr),3)
    mae_tr = round(mean_absolute_error(obs_tr, pred_tr),3)
    q2f2 = round(r2_score(obs_te, pred_te),3)
    q2f1 = round(1-(sum((obs_te - pred_te)**2)/sum((obs_te - np.mean(obs_tr))**2)),3)
    mae_te = round(mean_absolute_error(obs_te, pred_te),3)
    rmsec = round(np.sqrt(mean_squared_error(obs_tr, pred_tr)),3)
    rmsep = round(np.sqrt(mean_squared_error(obs_te, pred_te)),3)
    return r2, mae_tr, rmsec, q2f1, q2f2, rmsep, mae_te




def web_app(df1:pd.DataFrame, df2:pd.DataFrame, leng:int):
    train1=df1
    test1=df2
    avg = train1.mean()
    stdev = train1.std()
    train = (train1-avg)/stdev
    test = (test1-avg)/stdev
    new_col = train.columns.str.replace('[\[\]<>]', '()', regex=True)
    train.columns=new_col
    test.columns=new_col
    x_tr=train.iloc[:,:-1]
    x_te=test.iloc[:,:-1]
    y_tr=train.iloc[:,-1]
    y_te=test.iloc[:,-1]
    print(x_tr.shape)
    print(x_te.shape)
    print(y_tr.shape)
    print(y_te.shape)



    cvloo = LeaveOneOut()
    r2val=[]
    q2looval=[]
    rmsecval=[]
    maetrval=[]
    q2f1val=[]
    q2f2val=[]
    rmsepval=[]
    maeteval=[]
    with pd.ExcelWriter("Total_metrics.xlsx", engine="openpyxl", mode="w") as writer:
        for i in range(leng, len(x_tr.columns)+1, 1):
            new_fea = x_tr.iloc[:, 0:i].columns.tolist()
            new_xtr = x_tr[new_fea].copy()
            new_xte = x_te[new_fea].copy()

            # Model training
            nn_reg = MLPRegressor(random_state=0).fit(new_xtr, y_tr)
            rf_reg = RandomForestRegressor(random_state=0).fit(new_xtr, y_tr)
            svm_reg = SVR().fit(new_xtr, y_tr)
            xgb_reg = XGBRegressor(random_state=0).fit(new_xtr, y_tr)

            # Predictions
            nn_trp = pd.DataFrame(nn_reg.predict(new_xtr), columns=["MLP"], index=train.index)
            rf_trp = pd.DataFrame(rf_reg.predict(new_xtr), columns=["RF"], index=train.index)
            svm_trp = pd.DataFrame(svm_reg.predict(new_xtr), columns=["SVM"], index=train.index)
            xgb_trp = pd.DataFrame(xgb_reg.predict(new_xtr), columns=["XGB"], index=train.index)
            
            nn_tep = pd.DataFrame(nn_reg.predict(new_xte), columns=["MLP"], index=test.index)
            rf_tep = pd.DataFrame(rf_reg.predict(new_xte), columns=["RF"], index=test.index)
            svm_tep = pd.DataFrame(svm_reg.predict(new_xte), columns=["SVM"], index=test.index)
            xgb_tep = pd.DataFrame(xgb_reg.predict(new_xte), columns=["XGB"], index=test.index)
            
            x_trp = pd.concat([nn_trp, rf_trp, svm_trp, xgb_trp], axis=1)
            x_tep = pd.concat([nn_tep, rf_tep, svm_tep, xgb_tep], axis=1)
            
            # PLS model and metrics calculation
            pls2 = PLSRegression(n_components=1).fit(x_trp, y_tr)
            pls_trp2 = pls2.predict(x_trp)
            pls_tep2 = pls2.predict(x_tep)
            
            pred = cross_val_predict(pls2, x_trp, y_tr, cv=cvloo)
            q2loo = r2_score(y_tr, pred)
            
            # Collect metrics
            mlp_r2, mlp_mae_tr, mlp_rmsec, mlp_q2f1, mlp_q2f2, mlp_rmsep, mlp_mae_te = metrics(obs_tr=y_tr, 
                                                                                            pred_tr=nn_reg.predict(new_xtr), 
                                                                                            obs_te=y_te, pred_te=nn_reg.predict(new_xte), 
                                                                                            method="MLP")
            rf_r2, rf_mae_tr, rf_rmsec, rf_q2f1, rf_q2f2, rf_rmsep, rf_mae_te = metrics(obs_tr=y_tr, 
                                                                                        pred_tr=rf_reg.predict(new_xtr), 
                                                                                        obs_te=y_te, pred_te=rf_reg.predict(new_xte), 
                                                                                        method="RF")
            svm_r2, svm_mae_tr, svm_rmsec, svm_q2f1, svm_q2f2, svm_rmsep, svm_mae_te = metrics(obs_tr=y_tr, 
                                                                                            pred_tr=svm_reg.predict(new_xtr), 
                                                                                            obs_te=y_te, 
                                                                                            pred_te=svm_reg.predict(new_xte), 
                                                                                            method="SVM")
            xgb_r2, xgb_mae_tr, xgb_rmsec, xgb_q2f1, xgb_q2f2, xgb_rmsep, xgb_mae_te = metrics(obs_tr=y_tr, 
                                                                                            pred_tr=xgb_reg.predict(new_xtr), 
                                                                                            obs_te=y_te, 
                                                                                            pred_te=xgb_reg.predict(new_xte), 
                                                                                            method="XGB")
            pls_r2, pls_mae_tr, pls_rmsec, pls_q2f1, pls_q2f2, pls_rmsep, pls_mae_te = metrics(obs_tr=y_tr, 
                                                                                            pred_tr=pls_trp2, 
                                                                                            obs_te=y_te, pred_te=pls_tep2, method="PLS")

            met_dict = {
                "R2": [mlp_r2, rf_r2, svm_r2, xgb_r2, pls_r2],
                "MAE_TR": [mlp_mae_tr, rf_mae_tr, svm_mae_tr, xgb_mae_tr, pls_mae_tr],
                "RMSEC": [mlp_rmsec, rf_rmsec, svm_rmsec, xgb_rmsec, pls_rmsec],
                "Q2F1": [mlp_q2f1, rf_q2f1, svm_q2f1, xgb_q2f1, pls_q2f1],
                "Q2F2": [mlp_q2f2, rf_q2f2, svm_q2f2, xgb_q2f2, pls_q2f2],
                "RMSEP": [mlp_rmsep, rf_rmsep, svm_rmsep, xgb_rmsep, pls_rmsep],
                "MAE_TE": [mlp_mae_te, rf_mae_te, svm_mae_te, xgb_mae_te, pls_mae_te],
                "Q2LOO": ["--", "--", "--", "--", q2loo]
            }
            met_df = pd.DataFrame(met_dict, index=["MLP", "RF", "SVM", "XGB", "PLS"])
            r2val.append(pls_r2)
            q2looval.append(q2loo)
            rmsecval.append(pls_rmsec)
            maetrval.append(pls_mae_tr)
            q2f1val.append(pls_q2f1)
            q2f2val.append(pls_q2f2)
            rmsepval.append(pls_rmsep)
            maeteval.append(pls_mae_te)

            met_df.to_excel(writer, sheet_name=f"Descriptors_{i}")
            print("Models at Descriptors: ", i)

    pls_met_dict ={
        "R2":r2val,
        "Q2LOO":q2looval,
        "RMSEC":rmsecval,
        "MAE_Tr":maetrval,
        "Q2F1":q2f1val,
        "Q2F2":q2f2val,
        "RMSEP":rmsepval,
        "MAE_Te":maeteval
    }

    pls_met_df = pd.DataFrame(pls_met_dict, index=[str(i) for i in range(leng, len(x_tr.columns)+1, 1)])
    return pls_met_df