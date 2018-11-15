#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
yun
--------------------------------------
'''
import numpy as np
import pickle
import config as cfg
import csv
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from plot_results import LossHistory
from keras.optimizers import SGD
from data_generator import RatioDataGenerator,train_generator,SamplePairing,MixupGenerator,get_random_eraser,mixup
from data_generator import SPGenerator,MixupGenerator_prelab,extrapolationGenerator
from Unity import compute_eer,mat_2d_to_3d,prec_recall_fvalue,GetMel,GetTags,TagsToCategory,GetAllData,CreateFolder
from Unity import GetMel_Time_Shifting,GetMel_Time_Stretch,GetMel_bg,GetMel_Pitch_Shift
from CRNN_MODEL import CRNN_ATT
from keras.callbacks import TensorBoard
from CNN_MODEL import CNN_model
from CRNN_BGRU_AT_MODEL import CRNN_BGRU_AT
#from imblearn.over_sampling import SMOTE
from CRNN_MODEL_Multi import Multi_attention
from VGG_ATTLOC import VGG_AT
import pandas as pd
#Data generator
#use_RatioDataGenerator = True
use_RatioDataGenerator = False
use_mixup = True
#use_mixup = False
#use_mixup_erase = True
use_mixup_erase = False
#use_samplepairing = True
use_samplepairing = False
use_mixup_RatioDataGenerator = False
#use_mixup_RatioDataGenerator = True           
use_mixup_samplepairing  = False
#use_mixup_samplepairing = True 
#without_aug = True
without_aug = False

CreateFolder( cfg.dev_fe_fd )
CreateFolder( cfg.dev_fe_mel_fd )

####################################################


#GetMel(cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0)
#GetMel_bg(cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0)
#GetMel_Pitch_Shift(cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0) #pitch shift
#GetMel_Time_Shifting(cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0) #slightly shift the starting point of the audio
#GetMel_Time_Stretch(cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0) # slow down or speed up the audio sample
####################################################




# hyper-params超参数
n_labels = len( cfg.labels )


for fold in range(4):
    #range() 函数可创建一个整数列表，一般用在 for 循环中。
    fe_fd = cfg.dev_fe_mel_fd
#    agg_num = 11        # concatenate frames
#    hop = 1            # step_len
    act = 'relu'
#    n_hid = 500
    n_out = n_labels
    agg_num = 124        # concatenate frames
    hop = 5            # step_len
    n_hid = 1000
#    t_delay=33
    feadim=128
    num_classes = len( cfg.labels)
    pParameter = 0.1
    #mixup 
    tr_X, tr_y, _, te_X, te_y, te_na_list = GetAllData( fe_fd, agg_num, hop, fold )
#    tmp_X, tmp_y = mixup(tr_X, tr_y, alpha=1.5)
#    tr_X, tr_y = np.r_[tr_X, tmp_X], np.r_[tr_y, tmp_y]
    # smooth labels
#    prob = pParameter * 1.0 / (num_classes - 1)
#    for yIndex,dum1 in enumerate(tr_y):
#        for yprob,dum2 in enumerate(tr_y[yIndex, :]):
#            if (tr_y[yIndex, yprob] == 1):
#                tr_y[yIndex, yprob] = 1 - pParameter
#            else:
#                tr_y[yIndex, yprob] = prob
#    nsamples, nx, ny = tr_X.shape
#    d2_train_dataset = tr_X.reshape((nsamples,nx*ny))
#    sm = SMOTE(random_state=8)
#    tr_X, tr_y = sm.fit_sample(d2_train_dataset, tr_y.ravel())

    print(tr_X.shape)
    
    print('tr_y shape:', tr_y.shape)
    if use_mixup_RatioDataGenerator:
        tr_X_mr,tr_y_mr = mixup(tr_X, tr_y, alpha=0.3, debug=False)
        (_, n_time, n_freq) = tr_X_mr.shape    # (N, 33, 64)  
        tr_X_mr = tr_X_mr.reshape(tr_X_mr.shape[0], agg_num, n_freq,1)
        te_X = te_X.reshape(te_X.shape[0], agg_num, n_freq,1)
    
    if use_RatioDataGenerator:
        (_, n_time, n_freq) = tr_X.shape    # (N, 33, 64)  
        tr_X = tr_X.reshape(tr_X.shape[0], agg_num, n_freq)
        te_X = te_X.reshape(te_X.shape[0], agg_num, n_freq)
    elif use_mixup or use_samplepairing or without_aug or use_mixup_erase:     
        [batch_num, n_time, n_freq] = tr_X.shape
        tr_X = tr_X.reshape(tr_X.shape[0], agg_num, n_freq,1)
        te_X = te_X.reshape(te_X.shape[0], agg_num, n_freq,1)

    # Build model
    history = LossHistory()  
    
#    if BGAT:
#        tr_X = tr_X.reshape(tr_X.shape[0], agg_num, n_freq)
#        te_X = te_X.reshape(te_X.shape[0], agg_num, n_freq)
	
#########################################	
#choose modle	
#    model = CRNN_ATT(n_freq, n_time)  
    
#    model = VGG_AT(n_freq, n_time)
    
#    model = CRNN_BGRU_AT(n_freq, n_time)
    
    model = CNN_model( n_time,n_freq)
    
#    model = Multi_attention(n_time, n_freq, num_classes)
	
###################################    
    model.summary()  
    model.save_weights('crnn_05.h5')
    plot_model(model, to_file='model_samplepairing.png',show_shapes= True )
    
    
#    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])  
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
#    model.compile(optimizer = Adam(lr = 1e-4), loss = [focal_loss(gamma=2,alpha=0.25)], metrics = ['accuracy'])
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')  
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='binary_crossentropy', mode='min')  
#    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, epsilon=1e-4, mode='min')
    
#    tbCallBack = TensorBoard(log_dir='./Graph/mhatt1', histogram_freq=0, write_graph=True, write_images=True)
    batch_size = 44 
    if  use_RatioDataGenerator or use_mixup_RatioDataGenerator:
    # Data generator
        gen = RatioDataGenerator(batch_size=batch_size,type='train')
        model.fit_generator(generator=gen.generate([tr_X], [tr_y]), 
                        steps_per_epoch=100,    # 100 iters is called an 'epoch'
                        epochs=100,              # Maximum 'epoch' to train
                        verbose=1, 
                        callbacks=[earlyStopping,mcp_save,history], 
                        validation_data=(te_X, te_y)) 
        

    
    elif use_mixup:

        training_generator = MixupGenerator(tr_X, tr_y, batch_size=batch_size, alpha=0)()      
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=tr_X.shape[0] // batch_size,
                            validation_data=(te_X, te_y),
                            epochs=100, verbose=1,
                            callbacks=[earlyStopping, mcp_save, history])
        
    

    elif use_mixup_erase:
        datagen = ImageDataGenerator(
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        preprocessing_function=get_random_eraser(v_l=np.min(tr_X), v_h=np.max(tr_X)))
        
        training_generator = MixupGenerator_prelab(tr_X, tr_y, batch_size=batch_size, alpha=1.5, datagen=datagen)()
        model.fit_generator(generator=training_generator,
                            steps_per_epoch=tr_X.shape[0] // batch_size,
                            validation_data=(te_X, te_y),
                            epochs=100, verbose=1,
                            callbacks=[earlyStopping, mcp_save, history])
    elif without_aug:
        datagen = ImageDataGenerator(
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=False)
        model.fit_generator(datagen.flow(tr_X, tr_y, batch_size=batch_size),
                    steps_per_epoch=tr_X.shape[0] // batch_size,
                    validation_data=(te_X, te_y),
                    epochs=100, verbose=1,
                    callbacks=[earlyStopping, mcp_save, history])
    elif use_samplepairing:
        transform = None
        samplepairing_freq = 2 # switch on SamplePairing once in a while
        samplepairing_duration = 1 # switch on SamplePairing for how many epochs?
        samplepairing_end = 30 # end samplepairing after how many epochs
        for i in range(50):
            if not i % samplepairing_freq: 
                if i < samplepairing_end:
                    print('-------------------------------------------------')
                    print('SamplePairing switched on for {} epoch(s)'.format(samplepairing_duration))
                    X_train_aug, Y_train_aug = SamplePairing(tr_X, tr_y, 22)
                    training_generator = train_generator(X_train_aug, Y_train_aug, transform)
        
                    model.fit_generator(generator=training_generator,
                    steps_per_epoch= tr_X.shape[0] // batch_size,
                    validation_data=(te_X, te_y),
                    epochs=samplepairing_duration, verbose=1,
                    callbacks=[history])
                    print('-------------------------------------------------')
                    continue
            training_generator = MixupGenerator(tr_X, tr_y, batch_size=batch_size, alpha=0)()
            model.fit_generator(generator=training_generator,
            steps_per_epoch= tr_X.shape[0] // batch_size,
            validation_data=(te_X, te_y),
            epochs=1, verbose=1,
            callbacks=[history])   
            
#    elif use_samplepairing:
#        transform = None
#        X_train_aug, Y_train_aug = SamplePairing(tr_X, tr_y, 44)
#        training_generator = train_generator(X_train_aug, Y_train_aug, transform)
#
#        model.fit_generator(generator=training_generator,
#                            steps_per_epoch= tr_X.shape[0] // batch_size,
#                             validation_data=(te_X, te_y),
#                             epochs=100, verbose=1,
#                             callbacks=[earlyStopping, mcp_save, history])

    CreateFolder(cfg.dev_md_fd)
    
    
    thres = 0.4     # thres, tune to prec=recall, if smaller, make prec smaller

    n_labels = len( cfg.labels )
    
    gt_roll = []
    pred_roll = []
    #define results
    result_roll = []
    y_true_binary_c = []
    y_true_file_c = []
    y_true_binary_m = []
    y_true_file_m = []
    y_true_binary_f = []
    y_true_file_f = []
    y_true_binary_v = []
    y_true_file_v = []
    y_true_binary_p = []
    y_true_file_p = []
    y_true_binary_b = []
    y_true_file_b = []
    y_true_binary_o = []
    y_true_file_o = []
   # with open( cfg.dev_cv_csv_path, 'rb') as f:
    with open( cfg.dev_cv_csv_path, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
        # read one line
        for li in lis:
            ##
                na = li[1]
                fe_path = fe_fd + '\\' + na + '.f'
                info_path = cfg.dev_wav_fd + '\\' + na + '.csv'
                tags = GetTags( info_path )
                y = TagsToCategory( tags )
                X = pickle.load(open( fe_path, 'rb' ))
                    #X = pickle.load(open( fe_path, 'r' ))             
                    # aggregate data
                X3d = mat_2d_to_3d( X, agg_num, hop)
                if use_RatioDataGenerator:
                    X3d = X3d.reshape(X3d.shape[0], agg_num, n_freq)
                elif use_samplepairing or without_aug or  use_mixup or use_mixup_erase:
                    X3d = X3d.reshape(X3d.shape[0], agg_num, n_freq,1)
                p_y_pred = model.predict( X3d )
                p_y_pred = np.mean( p_y_pred, axis=0 )
                pred = np.zeros(n_labels)
                pred[ np.where(p_y_pred>thres) ] = 1
                pred_roll.append( pred )
                gt_roll.append( y )
                
            
                
                ind=0
                for la in cfg.labels:
                    if la=='S':
                        break
                    elif la=='c':
                        y_true_file_c.append(na)
                        y_true_binary_c.append(y[ind])
                    elif la=='m':
                        y_true_file_m.append(na)
                        y_true_binary_m.append(y[ind])
                    elif la=='f':
                        y_true_file_f.append(na)
                        y_true_binary_f.append(y[ind])
                    elif la=='v':
                        y_true_file_v.append(na)
                        y_true_binary_v.append(y[ind])
                    elif la=='p':
                        y_true_file_p.append(na)
                        y_true_binary_p.append(y[ind])
                    elif la=='b':
                        y_true_file_b.append(na)
                        y_true_binary_b.append(y[ind])
                    elif la=='o':
                        y_true_file_o.append(na)
                        y_true_binary_o.append(y[ind])
                    result=[na,la,p_y_pred[ind]]
                    result_roll.append(result)
                    ind=ind+1
                
                
                pred_roll.append( pred )
                gt_roll.append( y )
    
    pred_roll = np.array( pred_roll )
    gt_roll = np.array( gt_roll )
    #write csv for EER computation
   # csvfile=file('result.csv','wb')
    #csvfile = open('result.csv', 'wb')
    with open('result.csv', 'w',newline ='') as csvfile:
        writer=csv.writer(csvfile)
        #writer.writerow(['fn','label','score'])
        writer.writerows(result_roll)
        csvfile.close()
        
        # calculate prec, recall, fvalue
        prec, recall, fvalue = prec_recall_fvalue( pred_roll, gt_roll, thres )
        
        EER_c=compute_eer('result.csv', 'c', dict(zip(y_true_file_c, y_true_binary_c)))
        EER_m=compute_eer('result.csv', 'm', dict(zip(y_true_file_m, y_true_binary_m)))
        EER_f=compute_eer('result.csv', 'f', dict(zip(y_true_file_f, y_true_binary_f)))
        EER_v=compute_eer('result.csv', 'v', dict(zip(y_true_file_v, y_true_binary_v)))
        EER_p=compute_eer('result.csv', 'p', dict(zip(y_true_file_p, y_true_binary_p)))
        EER_b=compute_eer('result.csv', 'b', dict(zip(y_true_file_b, y_true_binary_b)))
        EER_o=compute_eer('result.csv', 'o', dict(zip(y_true_file_o, y_true_binary_o)))
        EER=(EER_c+EER_m+EER_v+EER_p+EER_f+EER_b+EER_o)/7.0    
        
        print(prec, recall, fvalue)
        print(EER_c,EER_m,EER_f,EER_v,EER_p,EER_b,EER_o)
        print(EER)
        
    history.loss_plot('epoch')   
#    training_vis(history)
    
################################################################
#FOR TEST
################################################################

CreateFolder( cfg.eva_fe_fd )
CreateFolder( cfg.eva_fe_mel_fd )



#GetMel( cfg.eva_wav_fd, cfg.eva_fe_mel_fd, n_delete=0 )


n_labels = len(cfg.labels)
confuse_mat = np.zeros((n_labels, n_labels))      # confusion matrix
frame_based_accs = []

fe_fd = cfg.eva_fe_mel_fd

def GetAllData_t( fe_fd, csv_file, agg_num, hop ):
    with open( csv_file, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    Xlist = []
        
    # read one line
    for li in lis:
        na = li[1]
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        X = pickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        Xlist.append( X3d )

    return np.concatenate( Xlist, axis=0 )
	

# do recognize and evaluation
thres = 0.4     # thres, tune to prec=recall
n_labels = len( cfg.labels )

CreateFolder( cfg.eva_results_fd )
txt_out_path = cfg.eva_results_fd+'/task4_results.csv'
fwrite = open( txt_out_path, 'w')


gt_roll = []
pred_roll = []
#define results
result_roll = []
y_true_binary_c = []
y_true_file_c = []
y_true_binary_m = []
y_true_file_m = []
y_true_binary_f = []
y_true_file_f = []
y_true_binary_v = []
y_true_file_v = []
y_true_binary_p = []
y_true_file_p = []
y_true_binary_b = []
y_true_file_b = []
y_true_binary_o = []
y_true_file_o = []


with open( cfg.eva_csv_path, 'r') as f:
    reader = csv.reader(f)
    lis = list(reader)

    # read one line
    for li in lis:
        na = li[1]
        full_na = na + '.16kHz.wav'
        
        # get features, tags
        fe_path = cfg.eva_fe_mel_fd + '\\' + na + '.f'
        info_path = cfg.eva_wav_fd + '\\' + na + '.csv'
        tags = GetTags( info_path )
        y = TagsToCategory( tags )
        X = pickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        if use_RatioDataGenerator:
            X3d = X3d.reshape(X3d.shape[0], agg_num, feadim)
        elif use_mixup_erase or use_samplepairing or without_aug or use_mixup:
            X3d = X3d.reshape(X3d.shape[0], agg_num, feadim,1)
        p_y_pred = model.predict( X3d )
        p_y_pred = np.mean( p_y_pred, axis=0 )     # shape:(n_label)
        pred = np.zeros(n_labels)
        pred[ np.where(p_y_pred>thres) ] = 1
        pred_roll.append( pred )
        gt_roll.append( y )  
        
        
        ind=0
        for la in cfg.labels:
            if la=='S':
                break
            elif la=='c':
                y_true_file_c.append(na)
                y_true_binary_c.append(y[ind])
            elif la=='m':
                y_true_file_m.append(na)
                y_true_binary_m.append(y[ind])
            elif la=='f':
                y_true_file_f.append(na)
                y_true_binary_f.append(y[ind])
            elif la=='v':
                y_true_file_v.append(na)
                y_true_binary_v.append(y[ind])
            elif la=='p':
                y_true_file_p.append(na)
                y_true_binary_p.append(y[ind])
            elif la=='b':
                y_true_file_b.append(na)
                y_true_binary_b.append(y[ind])
            elif la=='o':
                y_true_file_o.append(na)
                y_true_binary_o.append(y[ind])
            task4_results=[na,la,p_y_pred[ind]]
            result_roll.append(task4_results)
            ind=ind+1

        # write out data
            for j1 in range(7):
                fwrite.write( full_na + ',' + cfg.id_to_lb[j1] + ',' + str(p_y_pred[j1]) + '\n' )
                
        pred_roll.append( pred )
        gt_roll.append( y )            
fwrite.close()
print( "Write out to", txt_out_path, "successfully!")

pred_roll = np.array( pred_roll )
gt_roll = np.array( gt_roll )
    #write csv for EER computation
with open('task4_results_maxup.csv', 'w',newline ='') as csvfile:
	writer=csv.writer(csvfile)
	#writer.writerow(['fn','label','score'])
	writer.writerows(result_roll)
	csvfile.close()
	
	# calculate prec, recall, fvalue
	prec, recall, fvalue = prec_recall_fvalue( pred_roll, gt_roll, thres )
	
	EER_c=compute_eer('task4_results_maxup.csv', 'c', dict(zip(y_true_file_c, y_true_binary_c)))
	EER_m=compute_eer('task4_results_maxup.csv', 'm', dict(zip(y_true_file_m, y_true_binary_m)))
	EER_f=compute_eer('task4_results_maxup.csv', 'f', dict(zip(y_true_file_f, y_true_binary_f)))
	EER_v=compute_eer('task4_results_maxup.csv', 'v', dict(zip(y_true_file_v, y_true_binary_v)))
	EER_p=compute_eer('task4_results_maxup.csv', 'p', dict(zip(y_true_file_p, y_true_binary_p)))
	EER_b=compute_eer('task4_results_maxup.csv', 'b', dict(zip(y_true_file_b, y_true_binary_b)))
	EER_o=compute_eer('task4_results_maxup.csv', 'o', dict(zip(y_true_file_o, y_true_binary_o)))
	EER=(EER_c+EER_m+EER_v+EER_p+EER_f+EER_b+EER_o)/7.0    
	
	print(prec, recall, fvalue)
	print(EER_c,EER_m,EER_f,EER_v,EER_p,EER_b,EER_o)
	print(EER)          
# 创建DataFrame对象,保存结果
df = pd.DataFrame([prec, recall, fvalue,EER_c,EER_m,EER_f,EER_v,EER_p,EER_b,EER_o,EER],
                  columns=['result'], index=['pre','rec','fav','c','m','f','v','p','b','o','eer']) 
df.to_csv('D:/projects/CHiME-Home/DCASE2016_Task4-master/Result.csv')   