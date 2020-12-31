import random

import pandas as pd
from surveyweights import run_weighting_iteration, run_weighting_scheme, normalize_weights
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

# <editor-fold desc="Census">
GA_CENSUS = {'gender': {'Female': 0.511,
						'Male': 0.483,
						'Other': 0.006},
			 # Male-Female from 2010 US Census https://www.census.gov/prod/cen2010/briefs/c2010br-03.pdf, other from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5227946/
			 'race': {'White, not Hispanic': 0.520,
					  'Black, non-Hispanic': 0.326,
					  'Hispanic': 0.099,
					  'Other': 0.055},  # https://www.census.gov/quickfacts/GA
			 'education': {'Graduated from college': 0.183,
						   'Some college, no degree': 0.284,
						   'Completed graduate school': 0.091,
						   'Graduated from high school': 0.301,
						   'Less than high school': 0.141},
			 # https://statisticalatlas.com/state/Georgia/Educational-Attainment
			 'income': {'Under $15,000': 0.1376,
						'Between $15,000 and $49,999': 0.3524,
						'Between $50,000 and $74,999': 0.1801,
						'Between $75,000 and $99,999': 0.116,
						'Between $100,000 and $150,000': 0.1207,
						'Over $150,000': 0.0932},  # https://statisticalatlas.com/state/Georgia/Household-Income
			 'age': {'18-34': 0.278,
					 '35-54': 0.359,
					 '55-64': 0.167,
					 '65 or older': 0.196},
			 # https://www.statista.com/statistics/910774/georgia-population-share-age-group/
			 'vote2016': {'Donald Trump': 0.504,
						  'Hillary Clinton': 0.453,
						  'Other': 0.043},
			 'vote2020': {'Joe Biden': 0.495,
						  'Donald Trump': 0.493,
						  'Other': 0.012},
			 'loc_county': {'Fulton County, GA': 0.1,
							'Cobb County, GA': 0.072,
							'Gwinnett County, GA': 0.088,
							'DeKalb County, GA': 0.0715,
							'Another county in Georgia': 0.6685},
			 'gss_trust': {'Can trust': 0.331,
						   'Can\'t be too careful': 0.669},
			 # From GSS 2018 https://gssdataexplorer.norc.org/variables/441/vshow
			 'gss_bible': {'Word of God': 0.41,
						   'Inspired word': 0.46,
						   'Book of fables': 0.13},
			 # From GSS 2018 https://gssdataexplorer.norc.org/variables/364/vshow (Region=South)
			 'gss_spanking': {'Agree': 0.677, 'Disagree': 0.323},
			 # From GSS 2018 https://gssdataexplorer.norc.org/trends/Gender%20&%20Marriage?measure=spanking
			 'social_fb': {'No': 0.31, 'Yes': 0.69}}  # https://www.pewresearch.org/internet/fact-sheet/social-media/
# </editor-fold>

# <editor-fold desc="fox news weights">
# https://www.foxnews.com/elections/2020/general-results/voter-analysis?race=S&state=GA
FOX_NEWS_WEIGHTS = {'survey_method': {'Online': 0.288, 'IVR': 0.712},
                    'gender': {'Female': 0.524,
                               'Male': 0.47,
                               'Other': 0.006},
                    'race': {'White, not Hispanic': 0.63,
                             'Black, non-Hispanic': 0.29,
                             'Hispanic': 0.03,
                             'Other': 0.05},
                    'education': {'Completed graduate school': 0.15,
                                  'Graduated from college': 0.25,
                                  'Some college, no degree': 0.33,
                                  'Graduated from high school': 0.17,
                                  'Less than high school': 0.1},
                    'income': {'Under $15,000': 0.1376,
                               'Between $15,000 and $49,999': 0.3524,
                               'Between $50,000 and $74,999': 0.1801,
                               'Between $75,000 and $99,999': 0.116,
                               'Between $100,000 and $150,000': 0.1207,
                               'Over $150,000': 0.0932},
                    'age': {'18-34': 0.278,
                            '35-54': 0.359,
                            '55-64': 0.167,
                            '65 or older': 0.196},
                    'vote2016': {'Donald Trump': 0.504,
                                 'Hillary Clinton': 0.453,
                                 'Other': 0.043},
                    'vote2020': {'Joe Biden': 0.495,
                                 'Donald Trump': 0.493,
                                 'Other': 0.012},
                    'loc_county': {'Fulton County, GA': 0.1,
                                   'Cobb County, GA': 0.072,
                                   'Gwinnett County, GA': 0.088,
                                   'DeKalb County, GA': 0.0715,
                                   'Another county in Georgia': 0.6685},
                    'gss_trust': {'Can trust': 0.331,
                                  'Can\'t be too careful': 0.669 }, # From GSS 2018 https://gssdataexplorer.norc.org/variables/441/vshow
                    'gss_bible': {'Word of God': 0.41,
                                  'Inspired word': 0.46,
                                  'Book of fables': 0.13}, # From GSS 2018 https://gssdataexplorer.norc.org/variables/364/vshow (Region=South)
                    'gss_spanking': {'Agree': 0.677, 'Disagree': 0.323},  # From GSS 2018 https://gssdataexplorer.norc.org/trends/Gender%20&%20Marriage?measure=spanking
                    'social_fb': {'No': 0.31, 'Yes': 0.69}} # https://www.pewresearch.org/internet/fact-sheet/social-media/
# </editor-fold>


survey_root = pd.read_csv('src/responses_processed.csv')
weigh_on_root = ['gender', 'race', 'education', 'income', 'age', 'vote2016', 'vote2020', 'loc_county',
				 'gss_trust', 'gss_bible', 'gss_spanking', 'social_fb']

todo = list(range(int('111111111111', 2) + 1))
# todo = list(range(int('111', 2) + 1))

random.shuffle(todo)  # shuffle so prediction time is accurate


def calc_weights(weights_id_int, foxnews = False):
	weights_id = bin(weights_id_int)[2::].zfill(12)
	survey = survey_root.copy()

	weigh_on = []

	for index, switch in enumerate(weights_id):
		if switch == "1":
			weigh_on.append(weigh_on_root[index])

	# print(f"{weights_id_int}: {weigh_on}")

	if len(weigh_on) == 0:
		survey['weight'] = 1
	else:
		census = FOX_NEWS_WEIGHTS if foxnews else GA_CENSUS
		output = run_weighting_scheme(survey, iters=100, weigh_on=weigh_on, census=GA_CENSUS, verbose=0,
									  early_terminate=False)
		survey = output['final_df']

	survey.to_pickle(f"pkls/{weights_id}{foxnews}.pkl")


# num_cores = multiprocessing.cpu_count()-1  # save one for me!
num_cores = multiprocessing.cpu_count()
inputs = tqdm(todo)

if __name__ == "__main__":
	processed_list = Parallel(n_jobs=num_cores)(delayed(calc_weights)(i, False) for i in inputs)
	processed_list = Parallel(n_jobs=num_cores)(delayed(calc_weights)(i, True) for i in inputs)

