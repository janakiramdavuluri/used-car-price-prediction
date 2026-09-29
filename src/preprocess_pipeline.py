from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from src.preprocess import preprocessor
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw/CAR DETAILS FROM CAR DEKHO.csv'
FEATURES=['name','fuel','seller_type','transmission','owner','km_driven','vehicle_age']
def load_data():
    df=pd.read_csv(DATA).drop_duplicates().dropna().copy()
    df['vehicle_age']=2026-df['year']
    return df[FEATURES],df['selling_price']
def prepare_data(test_size=.2,random_state=42):
    X,y=load_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_state)
    transformer=clone(preprocessor)
    return transformer.fit_transform(X_train),transformer.transform(X_test),y_train,y_test,transformer
if __name__=='__main__':
    a,b,c,d,_=prepare_data();print('Train:',a.shape,'Test:',b.shape)
