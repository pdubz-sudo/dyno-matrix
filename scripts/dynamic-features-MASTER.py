from __future__ import division, print_function, absolute_import

import pandas as pd
import numpy as np
from matrix_factorization import *
from model_evaluations import *
from secrets import randbelow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import random

print('If a window graph comes up you must close it to continue.')
print('This will take several minutes...')

# Ingest pre-processed foursquare data
file_path = r'C:\Users\DS\Desktop\dynamic-features\data\cleaned_foursquare_df.pkl'
fs = pd.read_pickle(file_path)

# There are 2547 3-hour time intervals that have at least regional datapoint in 
# this dataset. If there are no events in the time intervals when grouping, 
# the new dataframe will skip over the no-event time interval. 
# Since I have many dynamic features that could possibly have different missing
# data, I will keep doing a "left" merge to the base 2547 rows 3-hour time 
# interval frame so I have one consistent dimension shape.
base_fs = fs.groupby(pd.Grouper(key='utcTimestamp', freq='3H'))['userId'].count().reset_index().rename(columns={'userId': 'count'})

#### Visitor Count: Number of unique users in region r at time interval t.
visitor = fs.groupby([pd.Grouper(key='utcTimestamp', freq='3H'),
                      'neighborhood'])['userId'].nunique().unstack(level=1).reset_index()
visitor_count = pd.merge(base_fs, visitor, on='utcTimestamp', how='left').drop('count', axis=1)
visitor_count = visitor_count.set_index('utcTimestamp')

# Make space for memory
del visitor

#### Observation Frequency: Number of check-ins in region r and its neighborhood at 
#### time interval t.
obs = fs.groupby([pd.Grouper(key='utcTimestamp', freq='3H'),
                  'neighborhood'])['userId'].count().unstack(level=1).reset_index()
obs_freq = pd.merge(base_fs, obs, on='utcTimestamp', how='left').drop('count', axis=1).set_index('utcTimestamp')

# Make space for memory
del obs

#### Visitor entropy: Diversity of visitors in a location with respect to their visits.
#### You have do merge the user ids total number of visits to the region and then divide the 
#### 2 columns to get the probability that they will be there at that time.
visits_r_t = fs.groupby(['neighborhood', pd.Grouper(key='utcTimestamp',
                                             freq='3H'), 'userId']).count().iloc[:,0]

# number of checkins per user
total_user_checkins = fs.groupby(['userId']).count().iloc[:,0]

# Join on user id and now you have userid number of checkins in that time interval and 
# a column that shows their total checkins.
fs_visits_r_t = pd.merge(visits_r_t.reset_index(), total_user_checkins.reset_index(),
                   on=['userId']).rename(columns={'venueId_x': 'reg_time_visits',
                                                  'venueId_y': 'total_visits'})

# Make room for memory
del visits_r_t, total_user_checkins

# Get the probability that user will be at region at time interval.
fs_visits_r_t['probability'] = fs_visits_r_t['reg_time_visits']/fs_visits_r_t['total_visits']
fs_visits_r_t['used_for_visitor_entropy'] = [-x*np.log2(x) for x in fs_visits_r_t['probability']]
# Sum all the individual user entropys for region r and time interval t.
incomplete_entropy_df = fs_visits_r_t.groupby(['neighborhood', 
                                    'utcTimestamp'])['used_for_visitor_entropy'].sum().unstack(level=0).reset_index()

visitor_entropy = pd.merge(base_fs, incomplete_entropy_df, 
                           how='left', on='utcTimestamp').drop('count', axis=1).set_index('utcTimestamp')

# Make space in memory.
del incomplete_entropy_df

#### Region Popularity: Assess popularity of region r at time interval t.
# Count how many total checkins at all time intervals 
all_checkins_at_time = fs.groupby(pd.Grouper(key='utcTimestamp', freq='3H'))['userId'].count()

# Divide regions checkins at time interval by total checkins at that time interval.
region_popularity = obs_freq.divide(all_checkins_at_time, axis=0)

del all_checkins_at_time

#### Visitor Ratio
# Groupby 'userId' and 'venueId'. Count to see how many times a user checked in that venueId. This will make a series.
user_venue_count = fs.groupby(['userId', 'venueId']).count().iloc[:,-1]
# Now that you have the count for venue checkins by each user you will re-group by 'userId' and apply a function 
# to each 'userId' grouping (sort descending and keep top 4) to get back the top 4 visited venues for that user.
user_grouping = user_venue_count.groupby(level=0, group_keys=False)  # When calling apply, add group keys argument to prevent key appearing twice IT.
user_top_venues = user_grouping.apply(lambda x: x.sort_values(ascending=False).head(4))
# We need the timestamp and borough of all the users top 4 checkins.
top_4_full_df = pd.merge(user_top_venues.reset_index(), fs, how='left', on=['userId', 'venueId'])

# Drop duplicates so I get the first visit of the users top 4 checkins.
first_visit = top_4_full_df.drop_duplicates(['userId', 'venueId'])
incomplete_visit_df = first_visit.groupby([pd.Grouper(key='utcTimestamp', freq='3H'), 'neighborhood']).count().iloc[:, 0].unstack(level=1)
##################### SANITY CHECK
# # All that is commented out, to make sure the time stamps are ascending since I will drop duplicates.
# # It seems like the timestamps are in order but just in case they are not I will order them because I
# # am going to remove the duplicates based on userId and venueId so it will only keep the first occurence
# # which gives up the first checkin timestamp.
# sort_df = top_4_full_df.set_index(['userId', 'venueId']).groupby(level=[0,1], group_keys=False).apply(lambda x: x.sort_values(by='utcTimestamp'))
# first_visit = sort_df.reset_index().drop_duplicates(['userId', 'venueId'])

# # Group by time stamp and neighborhood. Take a count and unpivot the column so that we have the 
# # have the first time occurences of the top 4 userid visits by venue. The rows containing all zeros are excluded so
# # this needs to be joined back to the base df so we have a row for each time interval.
# incomplete_visit_df = first_visit.groupby([pd.TimeGrouper(key='utcTimestamp', freq='3H'), 'borough']).count().iloc[:, 0].unstack(level=1)
######################
# Join
new_user_venue_checkins = pd.merge(base_fs, incomplete_visit_df.reset_index(), how='left', on='utcTimestamp').drop('count', axis=1)

# Calculate visitor ratio by dividing by observation frequency for that region at that time interval.
visitor_ratio = new_user_venue_checkins.set_index('utcTimestamp')/obs_freq

# Make space for memory.
del user_venue_count, user_grouping, user_top_venues, top_4_full_df
del first_visit,incomplete_visit_df, new_user_venue_checkins


#### All dynamic features done except for one of them. Concat them all together, fill NaN with 0, and do the matrix factorization to
#### make dynamic features.
# Concat features and fill NaN with 0.
original_features_df = pd.concat([visitor_count, 
           obs_freq,
           visitor_entropy,
           visitor_ratio,
           region_popularity], axis=1).fillna(0)

############################################################################################################################################
###########################################
# # Drop low activity hours (less than 20 obs freq and low activity neighborhoods)
# number_neighborhoods = obs_freq.columns.shape[0]
# neighborhood_names = obs_freq.columns
# low_activity_neighborhoods = neighborhood_names[obs_freq.fillna(0).sum()<300]
# low_activity_hour_indexes = obs_freq[obs_freq.fillna(0).sum(axis=1)<20].index
# original_features_df.drop(low_activity_neighborhoods, axis=1, inplace=True)
# original_features_df.drop(low_activity_hour_indexes, inplace=True)

# Drop low activity hours (less than 20 obs freq and low activity neighborhoods)
number_neighborhoods = obs_freq.columns.shape[0]
neighborhood_names = obs_freq.columns
low_activity_hour_indexes = obs_freq[obs_freq.fillna(0).sum(axis=1)<20].index
original_features_df.drop(low_activity_hour_indexes, inplace=True)


low_activity_neighborhoods = neighborhood_names[obs_freq.fillna(0).sum()<300]
original_features_df.drop(low_activity_neighborhoods, axis=1, inplace=True)
############################################################################################################################################
###########################################


columns = original_features_df.columns
index = original_features_df.index
K = 250  # Shape of second dimension of matrices. (time interval, K) x (K, regions)
dynamic_features, _ = dynamic_feature_estimation(original_features_df.fillna(0).values, K=K, alpha=0.001, beta=0.001, epochs=20, print_cost=True)

print('\n\n\n')

# Rebuild the df
full_dynamic_df = pd.DataFrame(dynamic_features, columns=columns, index=index)



###############################################################################
#### Prep Labels and find city to build model and predictions
# Ingest pre-processed one-hot Grand Larceny theft labels.
file_path = r'C:\Users\DS\Desktop\dynamic-features\data\cleaned_neighborhood_theft_labels.pkl'
theft_df = pd.read_pickle(file_path)

# The author said the crime labels are binary so I make them binary.
theft_df[theft_df>1]=1

## Sanity check
# np.where(np.asarray(theft_df)>1)

# Get the time indexes that both X and Y df's match on, 'inner', so
# I can start deciding which city I will run the model and can then actually
# build the model.
timestamp_indexes_keep = pd.merge(full_dynamic_df, theft_df,
                                  on='utcTimestamp').index

non_dynamic_df = original_features_df.loc[timestamp_indexes_keep]
dynamic_df = full_dynamic_df.loc[timestamp_indexes_keep]
theft_df = theft_df.loc[timestamp_indexes_keep]

del timestamp_indexes_keep, original_features_df, full_dynamic_df


# Calculate the percent that each neighborhood is labeled.
neighborhoods = theft_df.columns
percent_labeled = [int((sum(theft_df.loc[:,hood])/theft_df.shape[0])*100) for hood in neighborhoods]

for city, percent in list(zip(neighborhoods, percent_labeled)):
    print('{}%\t{}'.format(percent, city))
print('\n-------------------------------------------------------------------------------')
print("Above: percent of labeled theft data which is associated with it's neighborhood.")
print('-------------------------------------------------------------------------------')
print('\n\n')

del neighborhoods


def neighborhood_evaluation(neighborhood):
    '''The neighborhoods features are used to evaluate theft predictions with 
    SVM, LR, and Random Forest and to compare original sparse features, sparse scaled,  
    dynamic features, and dynamic scaled. The evaluation metrics are auc, f1, and accuracy.
    This function can be easily changed to return predictions, auc, f1, and 
    accuracy for all the differentcombinations of sparse, dynamic, 
    and scaled training data.
    
    Arguments:
    neighborhood -- str, neighborhood name located in columns dataframe.
    
    Return:
    None
    '''
    
    neighborhood_sparse_features = non_dynamic_df.loc[:, neighborhood]
    neighborhood_dynamic_features = dynamic_df.loc[:, neighborhood]
    neighborhood_labels = theft_df.loc[:, neighborhood]

    print('\n\nResults for {}:'.format(neighborhood))
    print('Frequency of thefts in {}: {}'.format(neighborhood, Counter(neighborhood_labels)))
    
    # The research paper uses a 50%/50% balanced set for the model
    # so I will make apply under-sampling techniques and drop some no-crime data.           
    theft_keep_indexes = np.where(neighborhood_labels==1)[0]
    non_theft_indexes = random.sample(population=list(np.where(neighborhood_labels==0)[0]), k=sum(neighborhood_labels==1))

    # Get the indexes of the rows that I will keep for model training.
    X_indexes = np.concatenate((non_theft_indexes, theft_keep_indexes), axis=0)
    # sort X_indexes
    X_indexes.sort()
    
    sparse_X = neighborhood_sparse_features.iloc[X_indexes]
    dynamic_X = neighborhood_dynamic_features.iloc[X_indexes]
    Y = neighborhood_labels.iloc[X_indexes]
    
    print('Frequency of thefts in {} after under-sampling: {}'.format(neighborhood, Counter(Y)))

    # Secured random selection which will be used to make
    # comparisons between sparse, dynamic, and scaled features.
    random_seed = randbelow(4294967000)
    X_train_sparse, X_test_sparse, Y_train, Y_test = train_test_split(sparse_X, Y, test_size=0.30, random_state=random_seed)
    X_train_dynamic, X_test_dynamic, Y_train, Y_test = train_test_split(dynamic_X, Y, test_size=0.30, random_state=random_seed)

    ## Making scaled sparse features (whole dataset)
    scalar = StandardScaler().fit(sparse_X)
    sparse_X_scaled = scalar.transform(sparse_X)

    # sparse scaled training and test set
    scalar = StandardScaler().fit(X_train_sparse)
    X_train_sparse_scaled = scalar.transform(X_train_sparse)
    X_test_sparse_scaled = scalar.transform(X_test_sparse)

    ### Making scaled dynamic features (whole dataset)
    scalar = StandardScaler().fit(dynamic_X)
    dynamic_X_scaled = scalar.transform(dynamic_X)

    # dynamic scaled training and test set
    scalar = StandardScaler().fit(X_train_dynamic)
    X_train_dynamic_scaled = scalar.transform(X_train_dynamic)
    X_test_dynamic_scaled = scalar.transform(X_test_dynamic)
    
    print('\nCross Validation Scores were taken from original sparse features and dynamic features to ')
    print('evaluate dynamic feature performance. The features were also scaled to check if scaling helped performance.')
    print('The order of features used for cross-validation are as follows:') 
    print('Sparse, Dynamic, Sparse scaled, Dynamic scaled')
    print('\n')

    ### Modelling
    ## SVM
    # Sparse vs dynamic
    svm_sparse_pred, svm_sparse_auc, svm_sparse_f1, accuracy_sparse_svm = svm_evaluation(sparse_X, Y, X_train_sparse, X_test_sparse, Y_train, Y_test)
    svm_pred, svm_auc, svm_f1, accuracy_svm =svm_evaluation(dynamic_X, Y, X_train_dynamic, X_test_dynamic, Y_train, Y_test)

    # Sparse scaled vs dynamic scaled
    svm_sparse_pred_s, svm_sparse_auc_s, svm_sparse_f1_s, accuracy_sparse_svm_s =svm_evaluation(sparse_X_scaled, Y, X_train_sparse_scaled, X_test_sparse_scaled, Y_train, Y_test)
    svm_pred_s, svm_auc_s, svm_f1_s, accuracy_svm_s = svm_evaluation(dynamic_X_scaled, Y, X_train_dynamic_scaled, X_test_dynamic_scaled, Y_train, Y_test)
    print('')

    ## Logistic Regression
    # Sparse vs dynamic
    lr_sparse_pred, lr_sparse_auc, lr_sparse_f1, accuracy_sparse_lr = lr_evaluation(sparse_X, Y, X_train_sparse, X_test_sparse, Y_train, Y_test)
    lr_pred, lr_auc, lr_f1, accuracy_lr =lr_evaluation(dynamic_X, Y, X_train_dynamic, X_test_dynamic, Y_train, Y_test)

    # Sparse scaled vs dynamic scaled
    lr_sparse_pred_s, lr_sparse_auc_s, lr_sparse_f1_s, accuracy_sparse_lr_s = lr_evaluation(sparse_X_scaled, Y, X_train_sparse_scaled, X_test_sparse_scaled, Y_train, Y_test)
    lr_pred_s, lr_auc_s, lr_f1_s, accuracy_lr_s = lr_evaluation(dynamic_X_scaled, Y, X_train_dynamic_scaled, X_test_dynamic_scaled, Y_train, Y_test)
    print('')

    ## Random Forest
    # Sparse vs dynamic
    rf_sparse_pred, rf_sparse_auc, rf_sparse_f1, accuracy_sparse_rf =RF_evaluation(sparse_X, Y, X_train_sparse, X_test_sparse, Y_train, Y_test)
    rf_pred, rf_auc, rf_f1, accuracy_rf =RF_evaluation(dynamic_X, Y, X_train_dynamic, X_test_dynamic, Y_train, Y_test)

    # Sparse scaled vs dynamic scaled
    rf_sparse_pred_s, rf_sparse_auc_s, rf_sparse_f1_s, accuracy_sparse_rf_s =RF_evaluation(sparse_X_scaled, Y, X_train_sparse_scaled, X_test_sparse_scaled, Y_train, Y_test)
    rf_pred_s, rf_auc_s, rf_f1_s, accuracy_rf_s = RF_evaluation(dynamic_X_scaled, Y, X_train_dynamic_scaled, X_test_dynamic_scaled, Y_train, Y_test)
    
    
    ## Neural Network
    # Need to get values and reshape Y_train to get this to work.
    Y_train = Y_train.values.reshape(-1,1)
    nn_sparse_pred_s, nn_sparse_auc_s, nn_sparse_f1_s, accuracy_sparse_nn_s = nn_evaluation(X_train_sparse_scaled, X_test_sparse_scaled, Y_train, Y_test, learning_rate=0.01, sigmoid_thresh=0.5, epochs=100, print_cost=False)
    nn_pred_s, nn_auc_s, nn_f1_s, accuracy_nn_s = nn_evaluation(X_train_dynamic_scaled, X_test_dynamic_scaled, Y_train, Y_test, learning_rate=0.01, sigmoid_thresh=0.5, epochs=100, print_cost=False)


    print('\n\n\n\nTest Set results.  *note: The cross-validation scores above give are a better indicator for evaluating')
    print('the overall comparison performace between sparse and dynamic features.')  
    print('')  
    print('Sparse Features (Original) \tDynmaic Features')
    print('')
    print('SVM')
    print(' non-scaled  scaled \t\tnon-scaled  scaled')
    print('auc: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(svm_sparse_auc, svm_sparse_auc_s, svm_auc, svm_auc_s))
    print(' f1: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(svm_sparse_f1, svm_sparse_f1_s, svm_f1, svm_f1_s))
    print('acc: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(accuracy_sparse_svm, accuracy_sparse_svm_s, accuracy_svm, accuracy_svm_s))

    print('\nLR')
    print(' non-scaled  scaled \t\tnon-scaled  scaled')
    print('auc: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(lr_sparse_auc, lr_sparse_auc_s, lr_auc, lr_auc_s))
    print(' f1: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(lr_sparse_f1, lr_f1_s, lr_f1,lr_f1_s))
    print('acc: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(accuracy_sparse_lr, accuracy_sparse_lr_s, accuracy_lr, accuracy_lr_s))

    print('\nRF')
    print('non-scaled   scaled \t\tnon-scaled  scaled')
    print('auc: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(rf_sparse_auc, rf_sparse_auc_s, rf_auc,rf_auc_s))
    print(' f1: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(rf_sparse_f1, rf_sparse_f1_s, rf_f1,rf_f1_s))
    print('acc: {:0.4f}, {:0.4f} \t\t    {:0.4f}, {:0.4f}'.format(accuracy_sparse_rf, accuracy_sparse_rf_s, accuracy_rf, accuracy_rf_s))
    
    print('\nNN. Both are scaled for gradient descent.')
    print('             scaled   \t\t            scaled')
    print('auc:         {:0.4f} \t\t            {:0.4f}'.format(nn_sparse_auc_s, nn_auc_s))
    print(' f1:         {:0.4f} \t\t            {:0.4f}'.format(nn_sparse_f1_s, nn_f1_s))
    print('acc:         {:0.4f} \t\t            {:0.4f}'.format(accuracy_sparse_nn_s, accuracy_nn_s))

    return None


neighborhood_evaluation('West Village')

print('\n\n\nMachine Learning Optimization info for matrices:')
print('Shape of features matrix (timestamp, features) : {}'.format(dynamic_df.shape))
print('Latent dimension K = {}'.format(K))