import os
import logging
import subprocess as sp
from pathlib import Path
from timeit import default_timer as timer

from cosmic.util import load_config, sysrun

logging.basicConfig(level=os.getenv('COSMIC_LOGLEVEL', 'INFO'), 
                    format='%(asctime)s %(levelname)8s: %(message)s')
logger = logging.getLogger(__name__)


def resolve_output_dir(config, runid, stream, year, month, output_type):
    return (config.BASE_OUTPUT_DIRPATH / runid / 
            f'{stream}.pp' / f'{output_type}_{year}{month:02}')


def check_access(runid):
    try:
        comp_proc = sysrun(f'moo ls moose:/crum/{runid}')
        logger.info(comp_proc.stdout)
        return None
    except sp.CalledProcessError as cpe:
        logger.error(f'Cannot access {runid}')
        logger.error(cpe)
        logger.error('===ERROR===')
        logger.error(cpe.stderr)
        logger.error('===ERROR===')
        if re.search('SSC_STORAGE_SYSTEM_UNAVAILABLE', cpe.stderr):
            logger.error('MASS down')
        raise


def write_stream_query(queries_dir, runid, stream, year, month, stashcodes, stream_info):
    if len(stashcodes) == 1:
        # No brackets surrounding one stashcode.
        stashcode_str = str(stashcode_str[0])
    else:
        # Brackets surrounding comma separated list of stashcodes.
        stashcode_str = '(' + ', '.join([str(s) for s in stashcodes]) + ')'

    output_name = stream_info['output_name']
    query_filepath = queries_dir / f'{runid}_{stream}_{year}{month:02}_{output_name}_select_query'
    logger.debug(f'  writing {query_filepath}')
    lines = []
    lines.append('begin')
    lines.append(f'  stash={stashcode_str}')
    lines.append(f'  year={year}')
    lines.append(f'  mon={month}')
    for element, element_val in stream_info['extra_elements']:
        lines.append(f'  {element}={element_val}')
    lines.append('end')
    with open(query_filepath, 'w') as fp:
        fp.write('\n'.join(lines) + '\n')
    return query_filepath


def run_moo_select(config, runid, stream, year, month, stream_info, query_filepath):
    output_dir = resolve_output_dir(config, runid, stream, year, month, stream_info)
    if not output_dir.exists():
        os.makedirs(output_dir)
    cmd = f'moo select {query_filepath} moose:/crum/{runid}/{stream}.pp/ {output_dir}'
    logger.debug(f'  {cmd}')
    try:
        comp_proc = sysrun(cmd)
        logger.info(comp_proc.stdout)
        return None
    except sp.CalledProcessError as cpe:
        logger.error('===ERROR===')
        logger.error(cpe.stderr)
        logger.error('===ERROR===')
        if re.search('ERROR_CLIENT_PATH_ALREADY_EXISTS', cpe.stderr):
            return cpe
        raise


def retrieve_from_MASS(config, queries_dir, runid, stream,
                       year, month, stashcodes, stream_info):
    query_filepath = write_stream_query(queries_dir, runid, stream, 
                                        year, month, stashcodes, stream_info)
    run_moo_select(config, runid, stream, year, month, stream_info, query_filepath)


def gen_years_months(start_year_month, end_year_month):
    curr_year_month = start_year_month
    years_months = []
    while curr_year_month <= end_year_month:
        years_months.append(curr_year_month)
        year = curr_year_month[0]
        month = curr_year_month[1] + 1
        if month == 13:
            month = 1
            year = curr_year_month[0] + 1
        curr_year_month = (year, month)
    return years_months


def main(config_filename):
    logger.info('retrieve_from_mass')
    config = load_config(config_filename)
    logger.debug(config.ACTIVE_RUNIDS)

    # Check have access to all runids.
    for runid in config.ACTIVE_RUNIDS:
        # check_access(runid)
        pass

    # Use config to download relevant data.
    for runid in config.ACTIVE_RUNIDS:
        mass_info = config.MASS_INFO[runid]
        queries_dir = Path('queries')
        queries_dir.mkdir(exist_ok=True)

        for stream, stream_info in mass_info['stream'].items():
            stashcodes = stream_info['stashcodes']
            years_months = gen_years_months(stream_info['start_year_month'], 
                                            stream_info['end_year_month'])
            for i, (year, month) in enumerate(years_months):
                N = len(years_months)
                logger.info(f'Retrieving {i+1}/{N} {runid}: {stream} {year}/{month} {stashcodes}')
                start = timer()
                retrieve_from_MASS(config, queries_dir, runid, stream,
                                   year, month, stashcodes, stream_info)
                end = timer()
                logger.info(f'Retrieved in {end - start:02f}s')
