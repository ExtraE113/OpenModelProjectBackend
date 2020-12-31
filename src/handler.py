import json
import pickle5 as pickle

import pandas as pd
import numpy as np
import scipy.stats as stat

from math import sqrt
from surveyweights import normalize_weights
import boto3
import os

# todo avx2?


def get_pkl(bucket_name, key):
	s3 = boto3.resource('s3')

	obj = s3.Object(

		bucket_name=bucket_name,

		key=key,

	)

	obj_body = obj.get()['Body'].read()
	data = pickle.loads(obj_body)

	return data



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
	# return (('Result {} {}{} (80% CI: {} to {}) (Weighted N={}) (raw_moe={}pts, margin={}pts, '
	# 		 'sigma={}pts) ({} {} likely to win)').format(dem_name,
	# 													  '+' if mean > 0 else '',
	# 													  mean,
	# 													  first,
	# 													  second,
	# 													  n,
	# 													  raw_moe,
	# 													  margin,
	# 													  sigma,
	# 													  dem_name,
	# 													  chance_pass))

	return {"headline": mean, "moe": raw_moe}


def hello(event, context):
	body = json.loads(event["body"])
	print(body)
	survey = get_pkl(os.environ["BUCKET"], body["weigh_on_code"] + body["exits"] + ".pkl")



	# %%

	survey['lv_index'] = 0
	survey.loc[(survey['lv_likely'] == 'Very likely'), 'lv_index'] += body["very likely"]
	survey.loc[(survey['lv_likely'] == 'Already voted'), 'lv_index'] += body["already voted"]
	survey.loc[(survey['lv_likely'] == 'Likely'), 'lv_index'] += body["likely"]
	survey.loc[(survey['lv_likely'] == 'Somewhat likely'), 'lv_index'] += body["somewhat likely"]
	survey.loc[(survey['lv_likely'] == 'Neither likely nor unlikely'), 'lv_index'] += body[
		"neither likely"]
	survey.loc[(survey['lv_likely'] == 'Somewhat unlikely'), 'lv_index'] += body["somewhat unlikely"]
	survey.loc[(survey['lv_likely'] == 'Unlikely'), 'lv_index'] += body["unlikely"]
	survey.loc[(survey['vote2020'] != 'Did not vote'), 'lv_index'] += body["2020 voted"]


	# https://www.pewresearch.org/methods/2016/01/07/measuring-the-likelihood-to-vote/
	perry_gallup_loadings = {7: 0.83, 6: 0.63, 5: 0.59, 4: 0.4, 3: 0.34, 2: 0.23, 1: 0.13, 0: 0.11}

	max_possible = max([body["very likely"],
						body["already voted"],
						body["likely"],
						body["somewhat likely"],
						body["neither likely"],
						body["somewhat unlikely"],
						body["unlikely"]]) + body["2020 voted"]

	survey['lv_index'] = survey['lv_index'].apply(
		lambda l: perry_gallup_loadings[int(np.round(l * 7.0 / max_possible))])
	survey['lv_weight'] = normalize_weights(survey['weight'] * survey['lv_index'])
	survey['lv_index'].value_counts()

	# %%

	# TODO: Turnout analysis (look at 2020 vote and people who were dropped)

	# %%

	# print('## OSSOFF vs. PERDUE - DEMOGRAPHIC WEIGHTS ##')
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
	ossoff_perdue_demo = print_result(**data)

	# %%

	# print('## OSSOFF vs. PERDUE - DEMOGRAPHIC WEIGHTS + 2020 RUNOFF LIKELY VOTERS ##')
	votes = survey_['vote_ossoff_perdue'].value_counts(normalize=True) * survey_.groupby('vote_ossoff_perdue')[
		'lv_weight'].mean() * 100
	votes = votes[options] * (100 / votes[options].sum())
	print(votes)
	data = calc_result(dem_vote=votes['Jon Ossoff'],
					   gop_vote=votes['David Perdue'],
					   n=lv_weighted_n)
	data['dem_name'] = 'Ossoff'
	ossoff_perdue_demo_likely = print_result(**data)

	# %%

	# print('## WARNOCK vs. LOEFFLER - DEMOGRAPHIC WEIGHTS ##')
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
	warnock_loeffler_demo = print_result(**data)

	# %%

	# print('## WARNOCK vs. LOEFFLER - DEMOGRAPHIC WEIGHTS + 2020 LIKELY VOTER ##')
	votes = survey_['vote_warnock_loeffler'].value_counts(normalize=True) * survey_.groupby('vote_warnock_loeffler')[
		'lv_weight'].mean() * 100
	votes = votes[options] * (100 / votes[options].sum())
	print(votes)
	data = calc_result(dem_vote=votes['Raphael Warnock'],
					   gop_vote=votes['Kelly Loeffler'],
					   n=lv_weighted_n)
	data['dem_name'] = 'Warnock'
	warnock_loeffler_demo_likely = print_result(**data)

	body = {
		"ossoff_perdue_demo": ossoff_perdue_demo,
		"ossoff_perdue_demo_likely": ossoff_perdue_demo_likely,
		"warnock_loeffler_demo": warnock_loeffler_demo,
		"warnock_loeffler_demo_likely": warnock_loeffler_demo_likely,
	}

	response = {
		"statusCode": 200,
		'headers': {
			'Access-Control-Allow-Headers': 'Content-Type',
			'Access-Control-Allow-Origin': '*',
			'Access-Control-Allow-Methods': 'POST'
		},
		"body": json.dumps(body)
	}

	return response
