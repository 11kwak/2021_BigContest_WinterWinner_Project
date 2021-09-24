from django.shortcuts import render
from .models import  *
from keras.models import load_model
import datetime
from dateutil.relativedelta import relativedelta
import os
# Create your views here.
def index_view(request):
    pred_p_price=None
    pred_period=None
    pred_dataset=None
    real_p_price=None
    lstm_epoches = None
    lstm_windowsize = None
    lstm_units = None
    lstm_batchsize = None
    lstm_ratio = None
    rmse = None
    gru_epoches = None
    gru_windowsize = None
    gru_units = None
    gru_batchsize = None
    gru_ratio = None

    windowsize_test =None
    test_result =None
    win_res =None
    filename = None
    modelname = None
    ratio = [[i, 10-i] for i in range(1, 10)]
    if request.method == "POST":
        file_path = r"_media"          #모델과 csv파일 저장의 기회는 단 한번뿐.
        if os.path.exists(file_path):
            for file in os.scandir(file_path):
                os.remove(file.path)


        fish_species = request.POST['fish_species']
        train_start = request.POST['train_start']
        train_end = request.POST['train_end']
        predict_period = request.POST['predict_period']
        select_model = request.POST['select_model']
        dataset ,_, df, windowsize_test = data_setting(fish_species,train_start,train_end)
        scacled_dataset , sc =  scaler(dataset)

        if select_model=='lstm':
            lstm_epoches = request.POST['lstm_epoches']
            lstm_windowsize = request.POST['lstm_windowsize']
            lstm_units = request.POST['lstm_units']
            lstm_batchsize = request.POST['lstm_batchsize']

            # print(windowsize_test)
            if int(lstm_windowsize) < windowsize_test:
                x, y, y_shape= split_xy(scacled_dataset,int(lstm_windowsize),int(predict_period))  ## 기간조정 예측기간이 윈도우보다 작으면 안됨
                model = model_setting(select_model,int(lstm_units),x,y)
                model= model_train(x,y,model,int(lstm_epoches),int(lstm_batchsize))
                y_pred = predict(model,sc,scacled_dataset,int(lstm_windowsize),y_shape)
                pred_p_price=y_pred[:,-1].tolist()
            else:
                test_result =1
                win_res = 'LSTM_WindowSize를 다시 조정하세요'
            
        elif select_model=='gru':
            gru_epoches = request.POST['gru_epoches']
            gru_windowsize = request.POST['gru_windowsize']
            gru_units = request.POST['gru_units']
            gru_batchsize = request.POST['gru_batchsize']
            if int(gru_windowsize) < windowsize_test:
                x, y, y_shape= split_xy(scacled_dataset,int(gru_windowsize),int(predict_period))  ## 기간조정 예측기간이 윈도우보다 작으면 안됨
                model = model_setting(select_model,int(gru_units),x,y)
                model= model_train(x,y,model,int(gru_epoches),int(gru_batchsize))
                y_pred = predict(model,sc,scacled_dataset,int(gru_windowsize),y_shape)
                pred_p_price=y_pred[:,-1].tolist()
            else:
                test_result = 1
                win_res = 'GRU_WindowSize를 다시 조정하세요'

        elif select_model=='lstm_gru':
            ensemble_ratio = request.POST['ratio']
            ensemble_ratio = ensemble_ratio.split("_")
            lstm_ratio = ensemble_ratio[0]
            gru_ratio = ensemble_ratio[1]
            lstm_epoches = request.POST['lstm_epoches']
            lstm_windowsize = request.POST['lstm_windowsize']
            lstm_units = request.POST['lstm_units']
            lstm_batchsize = request.POST['lstm_batchsize']
            gru_epoches = request.POST['gru_epoches']
            gru_windowsize = request.POST['gru_windowsize']
            gru_units = request.POST['gru_units']
            gru_batchsize = request.POST['gru_batchsize']
            if (int(lstm_windowsize) < windowsize_test)and(int(gru_windowsize) < windowsize_test):
                x, y, y_shape= split_xy(scacled_dataset,int(lstm_windowsize),int(predict_period))  ## 기간조정 예측기간이 윈도우보다 작으면 안됨
                model = model_setting('lstm',int(lstm_units),x,y)
                model = model_train(x,y,model,int(lstm_epoches),int(lstm_batchsize))
                y_pred = predict(model,sc,scacled_dataset,int(lstm_windowsize),y_shape)
                model2 = model_setting('gru',int(gru_units),x,y)
                model2 = model_train(x,y,model,int(gru_epoches),int(gru_batchsize))
                y_pred2 = predict(model2,sc,scacled_dataset,int(lstm_windowsize),y_shape)
                pred_p_price=ensemble(y_pred,y_pred2,int(lstm_ratio),int(gru_ratio))
            elif int(lstm_windowsize)>windowsize_test: 
                test_result =1
                win_res = 'LSTM_WindowSize를 다시 조정하세요'
            elif int(gru_windowsize)>windowsize_test: 
                test_result =1
                win_res = 'GRU_WindowSize를 다시 조정하세요'
        if test_result!= 1 :    
            predict_first_monday=datetime.datetime.strptime(f'{train_end}-1', r'%Y-%m-%d')
            predict_first_monday=predict_first_monday+relativedelta(months=1)
            while predict_first_monday.weekday()!=0:
                predict_first_monday=predict_first_monday+datetime.timedelta(days=1)

            j=predict_first_monday
            pred_period=[]

            for i in range(int(predict_period)):
                predict_day=predict_first_monday.strftime(r'%Y-%m-%d')
                if(predict_day=='2017-01-02'):
                    pred_period.append('2017-01-01')
                elif(predict_day=='2017-01-09'):
                    pred_period.append('2017-01-06')
                else:
                    pred_period.append(predict_day)
                predict_first_monday=predict_first_monday+datetime.timedelta(weeks=1)
            # print(b)
            # print(j)

            real_p_price=[]
            df=df[['p_price','reg_date']]
            df['reg_date'] = pd.to_datetime(df['reg_date'],format="%Y-%m-%d")
            df=df.set_index(df['reg_date'])
            for i in range(len(pred_period)):
                day=j.strftime(r'%Y-%m-%d')
                if(j<datetime.datetime.strptime('2020-12-31', r'%Y-%m-%d')):
                    if(day=='2017-01-02'):
                        day='2017-01-01'
                    elif(day=='2017-01-09'):
                        day='2017-01-06'
                    if((fish_species=='squid' and day=='2017-10-02') or (fish_species=='shrimp' and day=='2017-10-02')):
                        real_p_price.append('None')
                    else:
                        real_p_price.append(df.loc[day]['p_price'])
                else:
                    real_p_price.append('None')
                j=j+datetime.timedelta(weeks=1)

            if(pred_p_price):
                pred_dataset=[]
                for i in range(len(pred_p_price)):
                    pred_dataset.append((pred_p_price[i],pred_period[i],real_p_price[i]))
            real=[]
            pred=[]
            for i in range(len(real_p_price)):
                if(real_p_price[i]!='None'):
                    real.append(real_p_price[i])
                    pred.append(pred_p_price[i])
            if(len(real)!=0):
                rmse = RMSE(real,pred)
            else:
                rmse = '측정불가(None)'
            if(select_model== 'lstm_gru'):
                ensemble_ratio = lstm_ratio+':'+gru_ratio
            else:
                ensemble_ratio = None

            p_info = {
                'train_start':train_start,
                'train_end':train_end,
                'predict_period':predict_period,
                'select_model':select_model,
                'lstm_epoches':lstm_epoches,
                'lstm_windowsize':lstm_windowsize,
                'lstm_units':lstm_units,
                'lstm_batchsize':lstm_batchsize,
                'gru_epoches':gru_epoches,
                'gru_windowsize':gru_windowsize,
                'gru_units':gru_units,
                'gru_batchsize':gru_batchsize,
                'ensemble_ratio': ensemble_ratio
            }
            filename = save_csv(pred_p_price,pred_period,fish_species)
            if(select_model != 'lstm_gru'):
                modelname = save_single_model(model,select_model,fish_species,p_info,rmse)
            elif(select_model== 'lstm_gru'):
                modelname = save_ensemble_model(model,model2,fish_species,p_info,rmse)
        
            


    return render(request,'index.html', context={
        "ratio": ratio ,
        'pred_dataset': pred_dataset,
        'pred_p_price':pred_p_price,
        'pred_period':pred_period, 
        'real_p_price':real_p_price, 
        'rmse':rmse, 
        'lstm_epoches':lstm_epoches , 
        'lstm_windowsize':lstm_windowsize ,
        'lstm_units' : lstm_units,
        'lstm_batchsize' :lstm_batchsize  ,
        'lstm_ratio':lstm_ratio ,
        'gru_epoches': gru_epoches ,
        'gru_windowsize' :gru_windowsize , 
        'gru_units':gru_units ,
        'gru_batchsize': gru_batchsize,
        'gru_ratio': gru_ratio,
        'win_res':win_res,
        'filename' : filename,
        'modelname' : modelname,
        
        }
    )
      #c는 a+b, b는 기간, a는 예측값, d는 해당 기간의 실제 값




