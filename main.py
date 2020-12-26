# %%

import pandas as pd
import numpy as np
import scipy.stats as stat

from math import sqrt
from surveyweights import run_weighting_iteration, run_weighting_scheme, normalize_weights


def margin_of_error(n=None, sd=None, p=None, type='proportion', interval_size=0.95):
	z_lookup = {0.8: 1.28, 0.85: 1.44, 0.9: 1.65, 0.95: 1.96, 0.99: 2.58}
	if interval_size not in z_lookup.keys():
		raise ValueError('{} not a valid `interval_size` - must be {}'.format(interval_size,
																			  ', '.join(list(z_lookup.keys()))))
	if type == 'proportion':
		se = sqrt(p * (1 - p)) / sqrt(n)
	elif type == 'continuous':
		se = sd / sqrt(n)
	else:
		raise ValueError('{} not a valid `type` - must be proportion or continuous')

	z = z_lookup[interval_size]
	return se * z


def print_pct(pct, digits=0):
	pct = pct * 100
	pct = np.round(pct, digits)
	if pct >= 100:
		if digits == 0:
			val = '>99.0%'
		else:
			val = '>99.'
			for d in range(digits - 1):
				val += '9'
			val += '9%'
	elif pct <= 0:
		if digits == 0:
			val = '<0.1%'
		else:
			val = '<0.'
			for d in range(digits - 1):
				val += '0'
			val += '1%'
	else:
		val = '{}%'.format(pct)
	return val


def calc_result(dem_vote, gop_vote, n, interval=0.8):
	GENERAL_POLLING_ERROR = 3
	N_SIMS = 100000

	dem_moe = margin_of_error(n=n, p=dem_vote / 100, interval_size=interval)
	gop_moe = margin_of_error(n=n, p=gop_vote / 100, interval_size=interval)
	undecided = (100 - dem_vote - gop_vote) / 2

	dem_mean = dem_vote + undecided * 0.25
	dem_raw_moe = dem_moe * 100
	dem_allocate_undecided = undecided * 0.4
	dem_margin = dem_raw_moe + dem_allocate_undecided + GENERAL_POLLING_ERROR

	gop_mean = gop_vote + undecided * 0.25
	gop_raw_moe = gop_moe * 100
	gop_allocate_undecided = undecided * 0.4
	gop_margin = gop_raw_moe + gop_allocate_undecided + GENERAL_POLLING_ERROR

	cdf_value = 0.5 + 0.5 * interval
	normed_sigma = stat.norm.ppf(cdf_value)

	dem_sigma = dem_margin / 100 / normed_sigma
	dem_sims = np.random.normal(dem_mean / 100, dem_sigma, N_SIMS)

	gop_sigma = gop_margin / 100 / normed_sigma
	gop_sims = np.random.normal(gop_mean / 100, gop_sigma, N_SIMS)

	chance_pass = np.sum([sim[0] / 100 > sim[1] / 100 for sim in zip(dem_sims, gop_sims)]) / N_SIMS

	low, high = np.percentile(dem_sims - gop_sims, [20, 80]) * 100

	return {'mean': dem_mean - gop_mean, 'high': high, 'low': low, 'n': n,
			'raw_moe': dem_raw_moe + gop_raw_moe,
			'margin': (dem_margin + gop_margin) / 2,
			'sigma': (dem_sigma + gop_sigma) / 2,
			'chance_pass': chance_pass}


def print_result(mean, high, low, n, raw_moe, margin, sigma, chance_pass, dem_name, gop_name=None):
	mean = np.round(mean, 1)
	first = np.round(high, 1)
	second = np.round(low, 1)
	sigma = np.round(sigma * 100, 1)
	raw_moe = np.round(raw_moe, 1)
	margin = np.round(margin, 1)
	chance_pass = print_pct(chance_pass, 1)
	if second < first:
		_ = first
		first = second
		second = _
	if second > 100:
		second = 100
	if first < -100:
		first = -100
	print(('Result {} {}{} (80% CI: {} to {}) (Weighted N={}) (raw_moe={}pts, margin={}pts, '
		   'sigma={}pts) ({} {} likely to win)').format(dem_name,
														'+' if mean > 0 else '',
														mean,
														first,
														second,
														n,
														raw_moe,
														margin,
														sigma,
														dem_name,
														chance_pass))
	print('-')


# %%

survey = pd.read_csv('src/responses_processed.csv')

# %%

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

weigh_on = ['gender', 'race', 'education', 'income', 'age', 'vote2016', 'vote2020', 'loc_county',
			'gss_trust', 'gss_bible', 'gss_spanking', 'social_fb']
run_weighting_iteration(survey, weigh_on=weigh_on, census=GA_CENSUS)

# %%

output = run_weighting_scheme(survey, iters=35, weigh_on=weigh_on, census=GA_CENSUS, verbose=0, early_terminate=False)

# %%

survey['lv_index'] = 0
survey.loc[(survey['lv_likely'] == 'Very likely'), 'lv_index'] += 1
survey.loc[(survey['lv_likely'] == 'Already voted'), 'lv_index'] += 1
survey.loc[(survey['lv_likely'] == 'Likely'), 'lv_index'] += 0.7
survey.loc[(survey['lv_likely'] == 'Somewhat likely'), 'lv_index'] += 0.4
survey.loc[(survey['lv_likely'] == 'Neither likely nor unlikely'), 'lv_index'] += 0.2
survey.loc[(survey['lv_likely'] == 'Somewhat unlikely'), 'lv_index'] += 0.1
survey.loc[(survey['lv_likely'] == 'Unlikely'), 'lv_index'] += 0.05
survey.loc[(survey['vote2020'] != 'Did not vote'), 'lv_index'] += 0.5

# https://www.pewresearch.org/methods/2016/01/07/measuring-the-likelihood-to-vote/
perry_gallup_loadings = {7: 0.83, 6: 0.63, 5: 0.59, 4: 0.4, 3: 0.34, 2: 0.23, 1: 0.13, 0: 0.11}
survey['lv_index'] = survey['lv_index'].apply(lambda l: perry_gallup_loadings[int(np.round(l * 7 / 1.5))])
survey['lv_weight'] = normalize_weights(survey['weight'] * survey['lv_index'])
survey['lv_index'].value_counts()

# %%

# TODO: Turnout analysis (look at 2020 vote and people who were dropped)

# %%

print('## 2016 VOTE - DEMOGRAPHIC WEIGHTS ##')
options = ['Hillary Clinton', 'Donald Trump', 'Other', 'Did not vote']
survey_ = survey.loc[survey['vote2016'].isin(options)].copy()
survey_['weight'] = normalize_weights(survey_['weight'])
survey_['lv_weight'] = normalize_weights(survey_['lv_weight'])
weighted_n = int(np.round(survey_['weight'].apply(lambda w: 1 if w > 1 else w).sum()))
lv_weighted_n = int(np.round(survey_['lv_weight'].apply(lambda w: 1 if w > 1 else w).sum()))
votes = survey_['vote2016'].value_counts(normalize=True) * survey_.groupby('vote2016')['weight'].mean() * 100
votes = votes[options] * (100 / votes[options].sum())
print(votes)
data = calc_result(dem_vote=votes['Hillary Clinton'],
				   gop_vote=votes['Donald Trump'],
				   n=weighted_n)
data['dem_name'] = 'Clinton'
print_result(**data)

# %%

print('## 2016 VOTE - DEMOGRAPHIC WEIGHTS + 2020 RUNOFF LIKELY VOTERS ##')
votes = survey_['vote2016'].value_counts(normalize=True) * survey_.groupby('vote2016')['lv_weight'].mean() * 100
votes = votes[options] * (100 / votes[options].sum())
print(votes)
data = calc_result(dem_vote=votes['Hillary Clinton'],
				   gop_vote=votes['Donald Trump'],
				   n=lv_weighted_n)
data['dem_name'] = 'Clinton'
print_result(**data)

# %%

print('## 2020 PREZ VOTE - DEMOGRAPHIC WEIGHTS ##')
options = ['Joe Biden', 'Donald Trump', 'Other', 'Did not vote']
survey_ = survey.loc[survey['vote2020'].isin(options)].copy()
survey_['weight'] = normalize_weights(survey_['weight'])
survey_['lv_weight'] = normalize_weights(survey_['lv_weight'])
weighted_n = int(np.round(survey_['weight'].apply(lambda w: 1 if w > 1 else w).sum()))
lv_weighted_n = int(np.round(survey_['lv_weight'].apply(lambda w: 1 if w > 1 else w).sum()))
votes = survey_['vote2020'].value_counts(normalize=True) * survey_.groupby('vote2020')['weight'].mean() * 100
votes = votes[options] * (100 / votes[options].sum())
print(votes)
data = calc_result(dem_vote=votes['Joe Biden'],
				   gop_vote=votes['Donald Trump'],
				   n=weighted_n)
data['dem_name'] = 'Biden'
print_result(**data)

# %%

print('## 2020 PREZ VOTE - DEMOGRAPHIC WEIGHTS + 2020 RUNOFF LIKELY VOTERS ##')
votes = survey_['vote2020'].value_counts(normalize=True) * survey_.groupby('vote2020')['lv_weight'].mean() * 100
votes = votes[options] * (100 / votes[options].sum())
print(votes)
data = calc_result(dem_vote=votes['Joe Biden'],
				   gop_vote=votes['Donald Trump'],
				   n=lv_weighted_n)
data['dem_name'] = 'Biden'
print_result(**data)

# %%

print('## OSSOFF vs. PERDUE - DEMOGRAPHIC WEIGHTS ##')
options = ['Jon Ossoff', 'David Perdue', 'Undecided']
survey_ = survey.loc[survey['vote_ossoff_perdue'].isin(options)].copy()
survey_['weight'] = normalize_weights(survey_['weight'])
survey_['lv_weight'] = normalize_weights(survey_['lv_weight'])
weighted_n = int(np.round(survey_['weight'].apply(lambda w: 1 if w > 1 else w).sum()))
lv_weighted_n = int(np.round(survey_['lv_weight'].apply(lambda w: 1 if w > 1 else w).sum()))
votes = survey_['vote_ossoff_perdue'].value_counts(normalize=True) * survey_.groupby('vote_ossoff_perdue')[
	'weight'].mean() * 100
votes = votes[options] * (100 / votes[options].sum())
print(votes)
data = calc_result(dem_vote=votes['Jon Ossoff'],
				   gop_vote=votes['David Perdue'],
				   n=weighted_n)
data['dem_name'] = 'Ossoff'
print_result(**data)

# %%

print('## OSSOFF vs. PERDUE - DEMOGRAPHIC WEIGHTS + 2020 RUNOFF LIKELY VOTERS ##')
votes = survey_['vote_ossoff_perdue'].value_counts(normalize=True) * survey_.groupby('vote_ossoff_perdue')[
	'lv_weight'].mean() * 100
votes = votes[options] * (100 / votes[options].sum())
print(votes)
data = calc_result(dem_vote=votes['Jon Ossoff'],
				   gop_vote=votes['David Perdue'],
				   n=lv_weighted_n)
data['dem_name'] = 'Ossoff'
print_result(**data)

# %%

print('## WARNOCK vs. LOEFFLER - DEMOGRAPHIC WEIGHTS ##')
options = ['Raphael Warnock', 'Kelly Loeffler', 'Undecided']
survey_ = survey.loc[survey['vote_warnock_loeffler'].isin(options)].copy()
survey_['weight'] = normalize_weights(survey_['weight'])
survey_['lv_weight'] = normalize_weights(survey_['lv_weight'])
weighted_n = int(np.round(survey_['weight'].apply(lambda w: 1 if w > 1 else w).sum()))
lv_weighted_n = int(np.round(survey_['lv_weight'].apply(lambda w: 1 if w > 1 else w).sum()))
votes = survey_['vote_warnock_loeffler'].value_counts(normalize=True) * survey_.groupby('vote_warnock_loeffler')[
	'weight'].mean() * 100
votes = votes[options] * (100 / votes[options].sum())
print(votes)
data = calc_result(dem_vote=votes['Raphael Warnock'],
				   gop_vote=votes['Kelly Loeffler'],
				   n=weighted_n)
data['dem_name'] = 'Warnock'
print_result(**data)

# %%

print('## WARNOCK vs. LOEFFLER - DEMOGRAPHIC WEIGHTS + 2020 LIKELY VOTER ##')
votes = survey_['vote_warnock_loeffler'].value_counts(normalize=True) * survey_.groupby('vote_warnock_loeffler')[
	'lv_weight'].mean() * 100
votes = votes[options] * (100 / votes[options].sum())
print(votes)
data = calc_result(dem_vote=votes['Raphael Warnock'],
				   gop_vote=votes['Kelly Loeffler'],
				   n=lv_weighted_n)
data['dem_name'] = 'Warnock'
print_result(**data)
