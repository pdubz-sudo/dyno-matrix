### Historical Features
import numpy as np

def area_crime_dens(number_events, area):
	'''Takes in number of crime events over certain time interval over the last x amount
	of days and returns crime event density by area size. 

	Argument:
	number_events -- list, series, or array, number of events in that given time interval over certain amount of days.
	Each index of the list, or array, represents a day.
	area -- float, area size of region.

	Return:
	crime_dens_by_area -- float, normalized crime dens by area.'''
	crime_dens_by_area  = sum(number_events) / area
	
	return crime_dens_by_area

def pop_crime_dens(number_events, population):
	'''Takes in number of crime events over certain time interval over the last x amount
	of days and returns crime even density by population. 

	Argument:
	number_events -- list, series, or array, number of events in that given time interval over certain amount of days.
	Each index of the list, or array, represents a day.
	area -- int, population.

	Return:
	crime_dens_by_population -- float, normalized crime dens by population.'''
	crime_dens_by_population = sum(number_events) / population
	
	return crime_dens_by_population

## Geographic Features
def venue_cat_distribution(region_category_venues, total_venues):
    '''Calculates a certain venue category distribution in a region.
    
    Arguments:
    region_category_venues -- int, number of venues of certain category in a region.
    total_venues -- int, total number of venues in that region.
    
    Return:
    venue_dist -- venue category distribution.
    '''
    venue_dist = region_category_venues/total_venues
    
    return venue_dist

def venue_cat_dens(region_category_venues, area):
    '''Calculates a certain venue category distribution in a region.
    
    Arguments:
    region_category_venues -- int, number of venues of certain category in a region.
    area -- int, area of the region.
    
    Return:
    venue_dens -- venue category density.
    '''
    venue_dens = region_category_venues/area
    
    return venue_dens

def regional_diversity(venue_distribution):
    '''Quantify regional venue diversity, heterogeneity, using Shanon's Entropy.
    
    Arguments:
    venue_distribution -- series, array, or list. each index is a venue category value between 0 and 1.
    The sum of all venue categories in the series, or array, is 1.
    
    Return:
    shanons_entropy -- series or array, quanitified venue heterogeneity of the region.'''
    shanons_entropy = -1 * sum(venue_distribution * np.log(venue_distribution))

    return shanons_entropy



## Dynamic Features
def cosine_similarity(u, v):
    '''Calculates cosine similarity of vectors. 1 means similar, 0 means no relation, -1 means opposite.
    
    Arguments:
    u -- series, array, or list; vector.
    v -- series, array, or list; vector.
    
    Returns:
    cos_sim -- calcuated cosine similarity.'''
    
    dot = np.dot(u, v)
    norm_u = np.sqrt(sum(u**2))
    norm_v = np.sqrt(sum(v**2))
    
    cos_sim = dot / (norm_u * norm_v)
    
    return cos_sim
    
    return cos_sim

# def visitor_homogeneity(xxxx,xxxxx):
# 	xxxxxxxxxxx
# 	xxxxxxxxx

def region_popularity(column):
	'''Takes pandas series, or array, which the whole series consists of all the 
	regions with each index as an individual region and returns region popularity. 

	Argument:
	column -- series or array, number of checkins per region for each index.

	Return:
	region_pop -- series or array, popularity of each region.'''
	region_pop = column / sum(column)
	
	return region_pop

####def visitor_ratio(xxxx):
## new user visitors

def visitor_count(data):
	'''Absolute count of unique users in a region r at a time interval t.
	
	Argument:
	data -- series.

	Return:
	unique_count -- int, unique user count.'''
	unique_count = len(set(data))

	return unique_count

### def observation_frequency():
# number of checkins for given region at given time.