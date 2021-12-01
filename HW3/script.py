from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy.io import loadmat
import nibabel as nib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import cv2
# from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report
# import threading


# Utils functions
def smoothing(img):
    cimg = cv2.GaussianBlur(img,(3,3),0)
    _, cimg = cv2.threshold(cimg,127,255,cv2.THRESH_BINARY)
    return cimg


def masking(img, mask):
    masked = [np.where(mask!=0,img[:,:,:,i],mask) for i in range(img.shape[-1])]
    return np.array(masked)


# Global variables
labels = loadmat('./data/label.mat')
labels = labels['label'].flatten()


ses_test_path = './data/jobs/rsub-01_ses-test_task-fingerfootlips_bold.nii'
mask_ses_test_path = './data/jobs/meansub-01_ses-test_task-fingerfootlips_bold.nii'


ses_retest_path = './data/jobs/sub-01_ses-retest_task-fingerfootlips_bold.nii'
mask_ses_retest_path = './data/jobs/meansub-01_ses-retest_task-fingerfootlips_bold.nii'

paths = [(ses_test_path,mask_ses_test_path),(ses_retest_path,mask_ses_retest_path)]

def task(fmri_path, mask_path):
    if 'ses-test' in fmri_path:
        token = 'test'
    else:
        token = 'retest'
    
    print(f'++++++++++++++++++++++++++++++ {token} ++++++++++++++++++++++++++++++')
    
    # Load the data
    fmri = np.array(nib.load(fmri_path).get_fdata())
    mask_fmri = np.array(nib.load(mask_path).get_fdata())

    # Further processing the mask
    mask = smoothing(mask_fmri)
    # Save the mask
    np.save(f'./data/jobs/{token}_mask',mask)

    # Load the mask
    # mask = np.load(f'./data/jobs/token_mask.npy')


    # Mask the background
    masked = masking(fmri,mask)

    # Reshape the data to feed the model
    prev_shape = masked.shape
    masked_reshaped = np.reshape(masked,(prev_shape[0],-1))


    # Split the training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(masked_reshaped,labels,random_state=123,test_size=0.1)



    # Train the model without PCA
    # Refer to https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    # # this may take too long to fully execuate, 
    # # but feel free to uncomment and use it if computational power allows
    # kFolds = np.arange(2,20,1,dtype=int)

    # # Let's take some large k and tune the hyperparams 
    if token == 'retest':
        kFolds = [7,14,20]
    else:
        kFolds = [7,9,15]

    svc = SVC(random_state=123)
    r = np.arange(-1,1,1,dtype=float)
    
    # Uncoment the lines if you do not wish to tune the parameters on your own
    hyperparams = {
                    'kernel':['linear','rbf'],
                    'C':10**r,
                    'gamma':10**r,
                    'degree':np.arange(0,2,1,dtype=int),
                    # 'kernel':['linear'],
                    # 'C':[0.01],
                    # 'gamma':['auto'],
                    # 'degree':[0],
                }


    scores = []
    params = []
    clfs = []

    for k in kFolds:
        print(f'({token}) Baseline training starts with K-Fold {k} on model {svc} ... ')
        clf =  GridSearchCV(svc,hyperparams,n_jobs=-1,scoring='accuracy',cv=k)
        clf.fit(X_train,y_train)
        
        scores.append(clf.best_score_)
        params.append(clf.best_params_)
        clfs.append(clf.best_estimator_)
        
        print(f'({token}) {k}-Fold mean validation accuracy: {clf.best_score_}')
        
    print(f'({token}) Baseline training complete!')
    
    
    # Best K-Fold
    best_k = np.argmax(scores)

    # Best estimator
    best_score = scores[best_k]
    best_clf = clfs[best_k]
    best_params = params[best_k]
    print(f'({token}) Best mean validation accuracy: {best_score}')
    print(f'({token}) Best validation tuned parameters: {best_params}')

    # Score the testing dataset
    testing_score = best_clf.score(X_test,y_test)
    y_true, y_pred = y_test, best_clf.predict(X_test)
    
    print(f'({token})')
    print('============================= Classifcation  Report =============================')
    print(classification_report(y_true,y_pred))
    print('=================================================================================')

    # input('pause')

    print(f'({token}) PCA + Best Baseline Model training starts ...')
    # Create Pipeline with PCA
    pipe = Pipeline([
            ('pca',PCA(random_state=123)),
            ('svc',SVC(random_state=123)),
            ])

    # We use the best params obtained in the pure SVM as a baseline and tune the new model against PCA components only
    nComponents = np.arange(70,120,10)
    param_grid = {'pca__n_components':nComponents,
                    'svc__kernel':[best_params['kernel']],
                    'svc__C':[best_params['C']],
                    'svc__gamma':[best_params['gamma']],
                    'svc__degree':[best_params['degree']]}
    

    # Train the SVM model with PCA dimension reduction
    kfold = kFolds[best_k]
    clf_pca = GridSearchCV(pipe,param_grid,n_jobs=-1,scoring='accuracy',cv=kfold)
    clf_pca.fit(X_train,y_train)
    print(f'({token}) PCA + Best Baseline Model training complete!')
    best_score_pca = clf_pca.best_score_
    best_params_pca = clf_pca.best_params_
    # best_clf_pca = clf_pca.best_estimator_
    print(f'({token}) Best mean validation accuracy: {best_score_pca}')
    print(f'({token}) Best validation tuned parameters: {best_params_pca}')

    # Score the testing dataset
    # X_test_pca = best_clf_pca['pca'].transform(X_test)
    testing_score_pca = clf_pca.score(X_test,y_test)
    y_pca_true, y_pca_pred = y_test, clf_pca.predict(X_test)
    
    print(f'({token})')
    print('============================= Classifcation  Report =============================')
    print(classification_report(y_pca_true,y_pca_pred))
    print('=================================================================================')
    
    results = pd.DataFrame({
        'Method':['SVC','PCA+SVC'],
        'Best Params':[best_params,best_params_pca],
        'CV Accuracy':[best_score,best_score_pca],
        'Testing Accuracy':[testing_score,testing_score_pca],
    })
    
    results.to_csv(f'ses-{token}_results.csv',index=False)
    

if __name__ == '__main__':
    for fmri_path, mask_path in paths:
        # t = threading.Thread(target=task,args=(fmri_path,mask_path))
        # t.start()
        task(fmri_path,mask_path)
    
    # try out
    # task(ses_retest_path,mask_ses_retest_path)