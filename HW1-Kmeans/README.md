
Please refer to the Scripts kmeans.py and ExtraCredit.py for any explanation of each line of codes.
Key parameters are also discussed in the scripts. Please review the summary within each methods/functions, including variable types and short descriptions.



Discussion and Improvement

-kmeans.py 
This script has been written in a way that the algorithm can automatically identify the face cluster and separate it from the irrelevant classes.
However, the script can be further improve by using sys.argv to allow the user to specify certain key parameters in command line.
In addition, regarding the algorithm itself, it was pre-determined to use k=7 clusters while running KMeans. 
In the future, we may want to let the algorithm decide the optimal k value and tweak the some other hyperparameters independently.


-ExtraCredit.py
This script has overall involved more complex thoughts in its design. 
In preprocessing the image, the script pre-specified some values or rules, which could only be obtained empricially and mannually.
This process involved much more trials and errors and might be implemented automatically.
However, the most sophiscated algorithm we attempt to design, the more computational powers are required.
In this trade-off, the script does a great job in balancing the need of computational capacity and its functionality.


In conclusion, both scripts are capable of accomplishing the task but for sure there is some room to improve. 
In the next stage, we can 
	(1) train the algorithm with k from 2 to 100, for example, and let the algorithm decide which k is sufficient enough to segment the image
	(2) use other facial features, such as eyebows, to detect the face, rather than relying on skin color solely (eg. hand color vs face color)
	(3) introduce different colorspaces and utilize the unique attributes in separate channels to segregate similar colors
