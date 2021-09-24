import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.models import load_model
import datetime
from sklearn.metrics import mean_squared_error
from django.db import models
import csv
import os
import zipfile

def data_setting(name,start,end): #종류 이름(salmon, shirimp, squid), 학습데이터 시작날짜, 학습데이터 종료날짜
   
    df = pd.read_csv(f'./homp/data/{name}_result_0.4_16.7.csv',encoding='cp949') 

    df = df.fillna(0) 
    dfdata = df.copy()
    df['reg_date'] = pd.to_datetime(df['reg_date'],format="%Y-%m-%d")
    df=df.set_index(df['reg_date'])

    dataset = df.loc[f"{start}":f"{end}"].reset_index(drop=True)
    windowsize_test = len(dataset.index)
    use_col = dataset.columns[1:]

    dataset=dataset[use_col]
    dataset=dataset.values

    return dataset  , use_col, dfdata ,windowsize_test

def scaler(dataset):
    sc = MinMaxScaler(feature_range=(0,1))
    scaled_dataset = sc.fit_transform(dataset)
    return scaled_dataset, sc
    
def split_xy(dataset, time_steps, y_column): #학습할 dataset, 한 번에 학습할 windowsize, 예측할 날짜 크기
    x=[]
    y=[]
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number,:]
        
        x.append(tmp_x)
        y.append(tmp_y)

        x_array = np.array(x)
        y_array = np.array(y)

        y_shape = (y_array.shape[0],y_array.shape[1],y_array.shape[2])

        y_array = y_array.reshape(y_shape[0],y_shape[1]*y_shape[2])
        
    return x_array, y_array, y_shape

def load_models(modelname): # 모델 로드. 모델을 불러오려면 load_modeld을 해줍시다. 
    model = load_model(f"{modelname}.h5")
    model.summary()
    return model

def model_setting(type,unit,x,y):
    if type == 'lstm':
        type = LSTM
    elif type == 'gru':
        type = GRU

    model = Sequential()
    model.add(type(unit, input_shape=(x.shape[1],x.shape[2])))  # 활성화 함수 
    model.add(Dense(y.shape[1]))  
    model.summary()
    model.compile(optimizer='adam',loss='mse')
    
    return model
    
def model_train(x_train,y_train,model,epoch,batchsize):
    model.fit(x_train,y_train,epochs=epoch,batch_size=batchsize)
    return model

def predict(model,sc,dataset,window_size,y_shape):
    x_test = dataset[-window_size:] 
    x_test=x_test.reshape(1,x_test.shape[0],x_test.shape[1]) 
    y_pred = model.predict(x_test)  
    y_pred=y_pred.reshape(y_shape[1],y_shape[2])
    y_pred = sc.inverse_transform(y_pred)
    return y_pred

def ensemble(y_pred,y_pred2,a,b): #predict1, predict2, 앙상블하고자 하는 비율을 a:b로 적어주세요.(각각 1~9 사이)
    en=[]
    for i in range(len(y_pred)):
        en.append((y_pred[:,-1][i]*a*0.1+y_pred2[:,-1][i]*b*0.1))

    return en

def save_csv(data,date,p_name):
    datetime2=datetime.datetime.strptime(date[0], r'%Y-%m-%d')  
    datelist=[]
    df=pd.DataFrame(data)
    
    if(p_name=='salmon'):
        p_name='연어'
    elif(p_name=='squid'):
        p_name='오징어'
    else:
        p_name='새우'


    for i in range(len(df.index)):
        datetime3=datetime2.strftime(r'%Y-%m-%d')
        datelist.append(datetime3)
        datetime2=datetime2+datetime.timedelta(weeks=1)
    
    df.index=datelist
    df = df.reset_index()
    df.columns=[['날짜','예측가격']]
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    pathname = r'./_media/'
    filename = fr'{p_name}_예측결과_{now}.csv'
    totalfile=pathname+filename
    df.to_csv(totalfile,encoding='cp949')
    return filename


def save_single_model(model,name,p_name,p_info,rmse):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    now2= datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if(p_name=='salmon'):
        p_name='연어'
    elif(p_name=='squid'):
        p_name='오징어'
    else:
        p_name='새우'

    pathname = r'./_media/'
    filename = fr'{p_name}_{name}모델_{now}.h5'
    model.save(pathname+filename)

    txtfile = open(rf"./_media/{p_name}_{p_info['select_model']}모델_{now}.txt", 'w', encoding='utf-8')
    txtdata=now2+','+p_name
    for value in p_info.values():
        if value==None:
            value='NULL'
        txtdata += (','+value)
    if rmse=='측정불가(None)':
        rmse='NULL'
    else:
        rmse=str(rmse)
    txtdata += (','+rmse)
    
    txtfile.write('/* 시뮬레이터_실행_시간(DATE12),어종(c10),학습시작날짜(DATE6),학습종료날짜(DATE6),예측기간(i2),\r\
    모델종류(c8),LSTM_Epoches(i4),LSTM_Window_Size(i4),LSTM_Units(i4),LSTM_Batch_Size(i4),\r\
    GRU_Epoches(i4),GRU_Window_Size(i4),GRU_Unit(i4),GRU_Batch_Size(i4),앙상블비율(LSTM:GRU)(c3),\r\
    RMSE스코어(f20) */\r')
    txtfile.write(txtdata)
    txtfile.close()

    zipname=f"{p_name}_{p_info['select_model']}모델_{now}.zip"
    model_zip = zipfile.ZipFile(rf'./_media/{zipname}', 'w')
    for folder, subfolders, files in os.walk(r'./_media/'):
        for file in files:
            if file.endswith(('.h5','.txt')):
                model_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), r'./_media/'), compress_type = zipfile.ZIP_DEFLATED)



    return zipname


def save_ensemble_model(model1,model2,p_name,p_info,rmse):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    now2= datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if(p_name=='salmon'):
        p_name='연어'
    elif(p_name=='squid'):
        p_name='오징어'
    else:
        p_name='새우'

    pathname = r'./_media/'
    filename_LSTM = fr'{p_name}_LSTM모델_{now}.h5'
    filename_GRU = fr'{p_name}_GRU모델_{now}.h5'
    model1.save(pathname+filename_LSTM)
    model2.save(pathname+filename_GRU)

    txtfile = open(rf'./_media/{p_name}_앙상블모델_{now}.txt', 'w', encoding='utf-8')
    txtdata=now2+','+p_name
    for value in p_info.values():
        if value=='lstm_gru':
            value="ensemble"
        txtdata += (','+value)
    if rmse=='측정불가(None)':
        rmse='NULL'
    else:
        rmse=str(rmse)
    txtdata += (','+rmse)

    txtfile.write('/* 시뮬레이터_실행_시간(DATE12),어종(c10),학습시작날짜(DATE6),학습종료날짜(DATE6),예측기간(i2),\r\
    모델종류(c8),LSTM_Epoches(i4),LSTM_Window_Size(i4),LSTM_Units(i4),LSTM_Batch_Size(i4),\r\
    GRU_Epoches(i4),GRU_Window_Size(i4),GRU_Unit(i4),GRU_Batch_Size(i4),앙상블비율(LSTM:GRU)(c3),\r\
    RMSE스코어(f20) */\r')
    txtfile.write(txtdata)
    txtfile.close()

    zipname=f'{p_name}_앙상블모델_{now}.zip'
    ensemble_zip = zipfile.ZipFile(rf'./_media/{zipname}', 'w')
    for folder, subfolders, files in os.walk(r'./_media/'):
        for file in files:
            if file.endswith(('.h5','.txt')):
                ensemble_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder,file), r'./_media/'), compress_type = zipfile.ZIP_DEFLATED)

    ensemble_zip.close()

    return zipname
    
def RMSE(y_test,y_pred):
    return np.sqrt(mean_squared_error(y_test,y_pred))
