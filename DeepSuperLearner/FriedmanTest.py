# Friedman test
import sfa
from numpy.random import seed
from numpy.random import randn
from scipy.stats import friedmanchisquare
# seed the random number generator
import pandas as pd
from pingouin import friedman
# generate three independent samples
import os
import numpy
# compare samples
# interpret

import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp

alpha = 0.05


num_of_result_per_algo = 2
num_of_datasets = 1
num_of_algo = 1
parameter = 'AUC'


path_root = os.path.dirname(os.path.dirname(__file__))
path = path_root + r'\results.xlsx'
file = pd.read_excel(path)
file = file[['Dataset Name', 'Algorythm Name',parameter ]]

algos_names = pd.Series(file['Algorythm Name']).unique()
datasets_names = pd.Series(file['Dataset Name']).unique()

datasets_names_map = {}
for i in range(len(datasets_names)):
    datasets_names_map[datasets_names[i]] = i

algos_names_map = {}
for i in range(len(algos_names)):
    algos_names_map[algos_names[i]] = i



fridman_table = [[0]*len(algos_names)]*len(datasets_names)


for algo in algos_names:
    for dt in datasets_names:
        values = []
        for index,row in file.iterrows():
            if row['Dataset Name'] == dt and row['Algorythm Name'] == algo:
                values.append(row['AUC'])
        fridman_table[datasets_names_map[dt]][algos_names_map[algo]] = numpy.mean(values)

df = pd.DataFrame(fridman_table, columns=algos_names)


df_rank = df.rank(1, ascending=False, method='first')
# stat, p = friedmanchisquare('knn', 'dsl', 'jjj')
# print('Statistics=%.3f, p=%.3f' % (stat, p))

#result = friedmanchisquare(df['knn'], df['DSL'], df['DSL_Improve1'], df['DSL_Improve2'])

stat, p = friedmanchisquare(df_rank['knn'], df_rank['DSL'], df_rank['DSL_Improve1'], df_rank['DSL_Improve2'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
    #lm = sfa.ols('knn ~ C(knn)', data=df).fit()
    #anova = sa.stats.anova_lm(lm)
    #print(anova)
    #sp.posthoc_ttest(df, val_col='knn', group_col='DSL', p_adjust='holm')

print(df)

print(fridman_table)
#print(file)

