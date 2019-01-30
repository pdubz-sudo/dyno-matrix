Dynamic Features for Improving Crime Predictions in NYC with FourSquare Check-in Data



conda create --name env_name_here python=3.6 jupyter ipykernel numpy pandas==0.23.4

# https://www.lfd.uci.edu/~gohlke/pythonlibs/
# download GDAL, Fiona, pyproj, rtree, shapely for proper python version and windows bit
# pip install all those windows in that order
pip install GDAL-2.3.3-cp36-cp36m-win_amd64.whl
pip install Fiona-1.8.4-cp36-cp36m-win_amd64.whl
pip install pyproj-1.9.6-cp36-cp36m-win_amd64.whl
pip install Rtree-0.8.3-cp36-cp36m-win_amd64.whl
pip install Shapely-1.6.4.post1-cp36-cp36m-win_amd64.whl
pip install geopandas

conda install scikit-learn

# Lastly,
pip install matplotlib shapely pyshp descartes tensorflow