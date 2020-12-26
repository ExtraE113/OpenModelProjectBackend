import pandas as pd
from surveyweights import run_weighting_iteration, run_weighting_scheme, normalize_weights



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
			 'age': {'Under 65': 0.8128,
					 '65+': 0.1872},  # https://www.census.gov/quickfacts/GA
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

survey_root = pd.read_csv('src/responses_processed.csv')
weigh_on_root = ['gender', 'race', 'education', 'income', 'age', 'vote2016', 'vote2020', 'loc_county',
				 'gss_trust', 'gss_bible', 'gss_spanking', 'social_fb']

# for i in range(int('111111111111', 2) + 1):
for i in range(int('1111', 2) + 1):
	weights_id = bin(i)[2::].zfill(12)
	survey = survey_root.copy()

	weigh_on = []

	for index, switch in enumerate(weights_id):
		if switch == "1":
			weigh_on.append(weigh_on_root[index])

	print(weigh_on)

	run_weighting_iteration(survey, weigh_on=weigh_on, census=GA_CENSUS, verbose=False)

	output = run_weighting_scheme(survey, iters=35, weigh_on=weigh_on, census=GA_CENSUS, verbose=0, early_terminate=False)

	survey.to_pickle(f"pkls/{weights_id}.pkl")
