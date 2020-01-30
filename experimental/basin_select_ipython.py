# coding: utf-8
get_ipython().run_line_magic('run', 'contextfig.py')
get_ipython().run_line_magic('run', 'basin_diurnal_cycle_analysis.py')
SLIDING_SCALES
get_ipython().run_line_magic('run', 'basin_diurnal_cycle_analysis.py --scales sliding')
get_ipython().run_line_magic('run', 'basin_diurnal_cycle_analysis.py --scales sliding')
get_ipython().run_line_magic('run', 'basin_diurnal_cycle_analysis.py')
get_ipython().run_line_magic('run', 'basin_diurnal_cycle_analysis.py --scales sliding')
get_ipython().run_line_magic('run', 'basin_diurnal_cycle_analysis.py --scales sliding')
analysis.df_keys
(analysis.df_keys.method == 'harmonic') & (analysis.df_keys.type == 'phase_mag')
((analysis.df_keys.method == 'harmonic') & 
 (analysis.df_keys.type == 'phase_mag') & 
 (analysis.df_keys.analysis_order == 'basin_area_avg') &
 (analysis.df_keys.mode == 'amount'))
selector = ((analysis.df_keys.method == 'harmonic') & 
 (analysis.df_keys.type == 'phase_mag') & 
 (analysis.df_keys.analysis_order == 'basin_area_avg') &
 (analysis.df_keys.mode == 'amount'))
analysis.df_keys[selector)
analysis.df_keys[selector]
analysis.df_keys
selector = ((analysis.df_keys.method == 'harmonic') & 
 (analysis.df_keys.type == 'phase_mag') & 
 (analysis.df_keys.analysis_order == 'basin_area_avg') &
 (analysis.df_keys.mode == 'amount'))
selector.sum()
selector = ((analysis.df_keys.method == 'harmonic') & 
 (analysis.df_keys.type == 'phase_mag') & 
 (analysis.df_keys.analysis_order == 'basin_area_avg'))
selector.sum()
selector.mode
analysis.df_keys[selector]
analysis.df_keys[selector].mode
selector = ((analysis.df_keys.method == 'harmonic') & 
 (analysis.df_keys.type == 'phase_mag') & 
 (analysis.df_keys.analysis_order == 'basin_area_avg') &
 (analysis.df_keys['mode'] == 'amount'))
selector.sum()
analysis.df_keys[selector]
analysis.df_keys[selector].basin_scale
analysis.df_keys[selector].basin_scale.unique
analysis.df_keys[selector].basin_scale.unique()
analysis.df_keys[selector].group_by('dataset')
analysis.df_keys[selector].groupby('dataset')
analysis.df_keys[selector].groupby('dataset').iteritems()
analysis.df_keys[selector].groupby('dataset').mean()
analysis.df_keys[selector].groupby('dataset')
g = analysis.df_keys[selector].groupby('dataset')
cmorph = analysis.df_keys[selector & analysis.df_keys.dataset == 'cmorph']
cmorph = analysis.df_keys[selector & (analysis.df_keys.dataset == 'cmorph')]
cmorph
