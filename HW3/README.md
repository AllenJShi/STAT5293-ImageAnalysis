# Directory
* main directory
    * script.py
    * README.md
    * ses-test_results.csv
    * ses-retest_results.csv
    * data
        * label.mat
        * readme.txt
        * jobs
            * sub-01_ses-test_task-fingerfootlips_bold.nii
            * rsub-01_ses-test_task-fingerfootlips_bold.nii
            * meansub-01_ses-test_task-fingerfootlips_bold.nii
            * sub-01_ses-retest_task-fingerfootlips_bold.nii
            * rsub-01_ses-retest_task-fingerfootlips_bold.nii
            * meansub-01_ses-retest_task-fingerfootlips_bold.nii
            * test_realign.m
            * test_realign_job.m
            * retest_realign.m
            * retest_realign_job.m
            * test_mask.npy
            * retest_mask.npy

# Procedures
## Matlab Image Preprocessing
- Realignment of fMRI images using SPM. The Matlab script is provided under the data/jobs directory. The test_realign.m/test_realign_job.m and retest_realign.m/retest_realign_job.m scripts help prepare useful for further analysis.
- Before running the scripts, please make sure the raw image data are correctly placed into the directory, according to the directory hierarchy on the top of the document.
- The outputs will be generated and stored under the same folder as the raw data. Please be aware that if you changed the directory hierarchy, you should expect to manually modify all global path variables in the python script. Comments are included in the script.

## Python Analytics
### Masking Images & Feature Extraction
The first part of the python script works to process the signal within the brain images. Setting up a single threshold to exclude background noise assists the classification and enhances accuracy. This step constitutes the basic feature extraction effort, though Principal Component Analysis (PCA) will be introduced to reduce dimensionality in the later section.
### Classification
The baseline model, Support Vector Machine (SVM), is built to provide a sense of how accurate the model can be trained with limited data. Because the datasets are relatively small, the K-fold cross-validation method is deployed for model training. The number of folds, K, is regarded as a hyperparameter and the best one will be retained for future model training. 
There are four hyperparameters in the SVM model, namely kernel, C, gamma and degree. The tuning will be performed for varying K-fold groups and the best parameters will be retained for future model building.
Now, the best parameters are obtained and a more advanced model will be designed while inheriting the tuned hyperparameters and best K-fold parameter for training. The model is implemented as a pipeline with two steps, PCA for dimensionality reduction and SVM for classification. However, a new hyperparameter, n_components in PCA, is brought into the view. Fortunately, the prior model provided enough information regarding the other parameters and thus the new model can simply leverage the best parameters from the previous SVM model and only need to tune the PCA parameter in this stage.
Last but not least, the model will be evaluated based on mean cross-validation accuracy and testing accuracy. Further details of the model specifications and scoring results will be provided in the next section.

# Discussion
## Model Specifications & Evaluation
Note: The following table is saved under the main directory after executing the python script. But the best K-fold parameter will only appear in the terminal as the python program runs.

- ses-test

Model   | C     | degree | gamma          | kernel | n_components |
:---:   | :---: | :---:  | :---:          | :---:  | :---:        |
SVM     | 0.01  | 0      | 'auto' or 0.01 | linear | N/A          |
PCA+SVM | 0.01  | 0      | 'auto' or 0.01 | linear | 110          |
- ses-retest

Model   | C     | degree | gamma          | kernel | n_components |
:---:   | :---: | :---:  | :---:          | :---:  | :---:        |
SVM     | 0.01  | 0      | 'auto' or 0.01 | linear | N/A          |
PCA+SVM | 0.01  | 0      | 'auto' or 0.01 | linear | 100          |

- Best Model Performance

| Dataset       | CV Accuracy   | Testing Accuracy | K-fold        | Best Model   |
|:-------------:|:-------------:| :-------------:  |:-------------:|:------------:|
| ses-test      | 0.884848485   | 1                | 15            | SVM          |
| ses-retest    | 0.86038961    |0.789473684       | 17            | PCA+SVM      |

- Inferior Model Performance

| Dataset       | CV Accuracy   | Testing Accuracy | K-fold        |Inferior Model|
|:-------------:|:-------------:| :-------------:  |:-------------:|:------------:|
| ses-test      |0.866666667    | 1                | 15            | PCA+SVM      |
| ses-retest    | 0.848484848   |0.789473684       | 17            | SVM          |

Notice that the PCA+SVM models have the same parameters in their support vector classifiers as the SVM models because of the tunning hyperparameters setup above. One important observation is that the hyperparameters of SVC, except the kernel, have little contribution to the CV accuracy in this particular case. (In the terminal, after each round of K-fold, the best parameters will be printed out and they will appear the same as the model specification above.) 
In the ses-test dataset, the best model is SVM with C=0.01, degree=0, gamma=auto and kernel=linear. The best CV accuracy is obtained at K-fold=15. Compare to the PCA+SVM model, the SVM model achieves a higher CV accuracy likely because the variance distribution in the image is not evidently correlated. Thus, retaining more components (n=110) preserves more information for training. 
However, in the ses-retest dataset, it implied the opposite; at 17-fold, the PCA+SVM model (with the same hyperparameters as above) outperforms the SVM model and the tuned components is 100, which is lower than that trained on ses-test dataset. This can be regarded as firm proof of the variance difference between these two datasets. It is safe to argue that PCA is preferable in the dataset containing highly correlated data as a means of dimensionality reduction. 
In both datasets, a large K-fold is preferred. This is because the training sample is so limited and not enough for the model to achieve the desired sore. The train test split is also set to be 9:1, in exchange for more training accuracy. In the ses-test dataset, the testing accuracy is even as high as 100%. However, this is not a convincing sign of a perfect model, because the testing set is too small to be representative. In the next section, limitations and improvements will be discussed in more detail.


## Limitations and Improvements
- The model performance heavily depends on the datasets. Though both datasets come from the same field of practice and possess very similar information, the model needs to be re-tuned for better performance. However, one thing should be noticed that the model is trained only based on 184 samples every time. The lack of data may also contribute to this result. A simple remedy to it is to acquire and feed the model with a large volume of high-quality data.
- SVM might not be the best practice in this particular scenario because the labels represent 4 distinct classes whereas SVM is a binary classifier in its very nature. While running the script, it takes extensive time and computational power to train the model and tune model parameters. XGboost may be a good alternative as it has a faster training time and enhanced accuracy. 
- Ideally, tuning hyperparameters for the second model, PCA+SVM, should have been conducted independently. Due to the limited computational power, the task becomes unreasonably difficult and assuming some initial parameters from the prior best model becomes necessary. However, this also leads to speculation of how much SVM parameters contribute to the overall variation of accuracy in the PCA+SVM model. So far, since the SVM parameters are fixed, all variations are attributable to the number of principal components. But this does not mean no better result can be achieved.
- Threading is used in the script. This will reduce the overall running time. However, since the classification reports will be printed out in the terminal, using threading may cause confusion. Dataset names will be indicated at the beginning of each printout line. Feel free to uncomment the threading approach in the main function but be careful of reading the results in the terminal.
