from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'house_price.csv'
TARGET = 'TARGET(PRICE_IN_LACS)'
FEATURES = ['POSTED_BY','UNDER_CONSTRUCTION','RERA','BHK_NO.','BHK_OR_RK','SQUARE_FT','READY_TO_MOVE','RESALE','ADDRESS','LONGITUDE','LATITUDE']

def load_data():
    df = pd.read_csv(DATA)
    df = df.drop_duplicates().copy()
    df['POSTED_BY'] = df['POSTED_BY'].astype(str).str.strip().str.title()
    df['BHK_OR_RK'] = df['BHK_OR_RK'].astype(str).str.strip().str.upper()
    df['ADDRESS'] = df['ADDRESS'].astype(str).str.strip().str.lower()
    return df

def make_preprocessor():
    categorical = ['POSTED_BY','BHK_OR_RK','ADDRESS']
    numeric = [c for c in FEATURES if c not in categorical]
    return ColumnTransformer([
        ('categorical', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                                  ('onehot',OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=5))]), categorical),
        ('numeric', Pipeline([('imputer',SimpleImputer(strategy='median')),
                              ('scale',StandardScaler(with_mean=False))]), numeric),
    ], remainder='drop')

def metrics(y, pred):
    return {'mae_lakhs':float(mean_absolute_error(y,pred)),
            'median_absolute_error_lakhs':float(median_absolute_error(y,pred)),
            'rmse_lakhs':float(mean_squared_error(y,pred)**0.5),
            'r2':float(r2_score(y,pred)),
            'median_absolute_percentage_error_pct':float(np.median(np.abs((y-pred)/np.maximum(np.abs(y),1e-6)))*100)}

def main():
    df=load_data()
    X=df[FEATURES]; y=df[TARGET].astype(float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
    candidates={
      'Median baseline':None,
      'Ridge regression':TransformedTargetRegressor(regressor=Pipeline([('prep',make_preprocessor()),('model',Ridge(alpha=10.0))]),func=np.log1p,inverse_func=np.expm1),
      'Random forest':TransformedTargetRegressor(regressor=Pipeline([('prep',make_preprocessor()),('model',RandomForestRegressor(n_estimators=180,min_samples_leaf=2,max_features=0.85,n_jobs=-1,random_state=42))]),func=np.log1p,inverse_func=np.expm1),
      'Extra trees':TransformedTargetRegressor(regressor=Pipeline([('prep',make_preprocessor()),('model',ExtraTreesRegressor(n_estimators=220,min_samples_leaf=2,max_features=0.9,n_jobs=-1,random_state=42))]),func=np.log1p,inverse_func=np.expm1),
    }
    results={}
    fitted={}
    for name,model in candidates.items():
        pred=np.full(len(y_test),y_train.median()) if model is None else None
        if model is not None:
            model.fit(X_train,y_train); pred=model.predict(X_test); fitted[name]=model
        results[name]=metrics(y_test,pred)
        print(name,results[name],flush=True)
    winner=min((n for n in fitted),key=lambda n:results[n]['mae_lakhs'])
    best=fitted[winner]
    joblib.dump(best,ROOT/'models'/'house_price_pipeline.joblib')
    report={'data_rows_raw':int(len(pd.read_csv(DATA))),'data_rows_after_duplicate_removal':int(len(df)),
            'duplicate_rows_removed':int(len(pd.read_csv(DATA))-len(df)), 'features':FEATURES,'target':TARGET,
            'train_rows':len(X_train),'test_rows':len(X_test),'random_state':42,'test_size':0.20,
            'winner':winner,'models':results,'target_median_lakhs':float(y.median()),
            'target_mean_lakhs':float(y.mean()),'target_min_lakhs':float(y.min()),'target_max_lakhs':float(y.max()),
            'area_median_sqft':float(df['SQUARE_FT'].median()),'area_p99_sqft':float(df['SQUARE_FT'].quantile(.99)),
            'area_max_sqft':float(df['SQUARE_FT'].max()), 'missing_cells':int(df.isna().sum().sum()),
            'unique_addresses':int(df.ADDRESS.nunique()),'sklearn_version':__import__('sklearn').__version__}
    (ROOT/'reports'/'model_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('WINNER',winner,'REPORT',ROOT/'reports'/'model_metrics.json',flush=True)
if __name__=='__main__': main()


