from pathlib import Path

BASE_OUTPUT_DIRPATH = Path('/gws/nopw/j04/cosmic/mmuetz/data')

LS_RAINFALL = 4203
LS_SNOWFALL = 4204

TOTAL_RAINFALL = 5214
TOTAL_SNOWFALL = 5215
TOTAL_PPT = 5216

AP9_LS_RAIN_SNOW_ONLY = {
    'stashcodes': [LS_RAINFALL, LS_SNOWFALL],
    'extra_elements': {},
    'output_name': 'precip',
}

AP9_TOTAL_RAIN_SNOW_ONLY = {
    'stashcodes': [TOTAL_RAINFALL, TOTAL_SNOWFALL],
    'extra_elements': {},
    'output_name': 'precip',
}

AP9_TOTAL_PPT_ONLY = {
    'stashcodes': [TOTAL_PPT],
    'extra_elements': {},
    'output_name': 'precip',
}


