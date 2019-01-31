# Dynamic Features for Improving Crime Predictions in NYC with FourSquare Check-in Data

I wanted to try out the implementation of the following research paper https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-018-0171-7.

Specifically, I wanted to see if I could improve the AUC metric of 4 classifier models by applying collaborative filtering,
matrix factorization, to a sparse dataset.

I created a custom matrix factorization algorithm function which can be found in the matrix_factorization.py file in the scripts directory.

Models
1-layer fully connected Neural Network in TensorFlow
Logistic Regression - sklearn
SVM - sklearn
Random Forest - sklearn

I used the following data
- FourSquare check-in data from Kaggle https://www.kaggle.com/chetanism/foursquare-nyc-and-tokyo-checkin-dataset
- NYC GIS shapefiles https://geo.nyu.edu/catalog/nyu-2451-34563
- NYPD complaint Data Historic https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i

### Quick wrap up of what I did
I took FourSquare check-in and theft crime coordinates and placed them in NYC neighborhoods using GIS shapefiles.
Cleaned and pre-processed the data.
Created dynamic features from the research paper. At first it is sparse.
Apply collaborative filtering to turn the sparse matrix to a dynamic dense matrix.
Compare sparse vs dynamic matrix performance in the 4 ML models.
The whole process can be found in the dynamic-features-MASTER.ipynb notebook. The parameter settings in the
notebook are the ones that performed the best.
The script dynamic-features-MAIN.py runs the script with cleaned data files in pickle format. To test how the script 
performs on other NYC neighborhoods, change the name of the neighborhood in the last function at the end of the script.

### Results
Collaborative filtering improved AUC performace up to 5% after taking an after of 14 runs of 5-kfold cross-validation 
for the West Village neighborhood in NYC. 
  

Installation Instrutions
GeoPandas is tricky to install. This is the safest bet to installing and having the whole jupyter notebook script work.

conda create --name env_name_here python=3.6 jupyter ipykernel numpy pandas==0.23.4

download GDAL, Fiona, pyproj, rtree, shapely for python 3.6 version and proper windows bit https://www.lfd.uci.edu/~gohlke/pythonlibs/

pip install all those windows in that order

pip install GDAL-2.3.3-cp36-cp36m-win_amd64.whl

pip install Fiona-1.8.4-cp36-cp36m-win_amd64.whl

pip install pyproj-1.9.6-cp36-cp36m-win_amd64.whl

pip install Rtree-0.8.3-cp36-cp36m-win_amd64.whl

pip install Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl

pip install geopandas

conda install scikit-learn

pip install matplotlib shapely pyshp descartes tensorflow