"""remakefile for downloading data -- one task per year.

Downloads GMP IMERG (late/final) daily data for Asia.

Will not download same file twice"""
import datetime as dt
from pathlib import Path
from timeit import default_timer as timer

import requests

from remake.version import VERSION
assert VERSION[0] >= 1 or VERSION[1] >= 5, 'remake version must by >= 0.5.0'

from remake import Remake, TaskRule

from cosmic.config import PATHS

IMERG_FINAL_30MIN_DIR = PATHS['gcosmic'] / Path('mmuetz/data/GPM_IMERG_final/30min')

FINAL_30MIN_URL_TPL = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.06/{year}/{doy}/{filename}?precipitationCal[0:0][2350:3329][889:1479],time,lon[2350:3329],lat[889:1479]'
FINAL_30MIN_FILENAME_TPL = '3B-HHR.MS.MRG.3IMERG.{datestr}-S{start_time}-E{end_time}.{minutes}.V06B.HDF5.nc4'

YEARS = range(2000, 2021)


class GpmDatetime(dt.datetime):
    """Useful fmt_methods added to datetime class"""
    def fmt_date(self):
        return self.strftime('%Y%m%d')

    def fmt_year(self):
        return self.strftime('%Y')

    def fmt_month(self):
        return self.strftime('%m')

    def fmt_doy(self):
        return self.strftime('%j')

    def fmt_time(self):
        return self.strftime('%H%M%S')

    def fmt_minutes(self):
        minutes = self.hour * 60 + self.minute
        return f'{minutes:04d}'


def get_from_gpm(url, filename):
    """Retrive from NASA GPM IMERG data repository

    note: $HOME/.netrc must be set!
    https://disc.gsfc.nasa.gov/data-access#python-requests
    """
    result = requests.get(url)
    try:
        result.raise_for_status()
        with open(filename,'wb') as f:
            f.write(result.content)
    except:
        print('requests.get() returned an error code ' + str(result.status_code))
        raise



def gen_dates_urls_filenames(filename_tpl, url_tpl, start_date, end_date):
    """Generates dates, urls and filenames in 30min intervals"""
    curr_date = start_date
    while curr_date < end_date:
        next_date = curr_date + dt.timedelta(minutes=30)
        filename =  filename_tpl.format(datestr=curr_date.fmt_date(),
                                        start_time=curr_date.fmt_time(),
                                        end_time=(next_date - dt.timedelta(seconds=1)).fmt_time(),
                                        minutes=curr_date.fmt_minutes())
        url = url_tpl.format(year=curr_date.fmt_year(),
                             doy=curr_date.fmt_doy(),
                             filename=filename)
        yield curr_date, url, filename
        curr_date = next_date


downloader = Remake()


class GpmImerg30MinDownload(TaskRule):
    rule_inputs = {}
    rule_outputs = {'output_filenames': IMERG_FINAL_30MIN_DIR / '{year}' / 'download.{year}.done'}
    var_matrix = {'year': YEARS}

    def rule_run(self):
        year = self.year
        if year == 2000:
            start_date = GpmDatetime(year, 6, 1)
        else:
            start_date = GpmDatetime(year, 1, 1)
        if year == 2020:
            # end_date = GpmDatetime(2020, 3, 3)
            end_date = GpmDatetime(year, 9, 1)
        else:
            end_date = GpmDatetime(year + 1, 1, 1)

        outputs = {}
        filename_tpl = FINAL_30MIN_FILENAME_TPL
        url_tpl = FINAL_30MIN_URL_TPL
        dates_urls_filenames = list(gen_dates_urls_filenames(filename_tpl,
                                                             url_tpl,
                                                             start_date,
                                                             end_date))
        all_filenames = []
        for i, (date, url, filename) in enumerate(dates_urls_filenames):
            output_filename = self.outputs['output_filenames'].parent / f'{date.timetuple().tm_yday}' / filename
            # output_filename = Path(IMERG_FINAL_DIR / date.fmt_year() / filename)
            if not output_filename.exists():
                outputs[url] = output_filename
            all_filenames.append(str(output_filename))

        for url, output_filename in outputs.items():
            print(url, output_filename)
            start = timer()
            tmp_filename = Path(output_filename.parent / ('.tmp.gpm_download.' + output_filename.name))
            tmp_filename.parent.mkdir(exist_ok=True)
            get_from_gpm(url, tmp_filename)
            assert tmp_filename.exists()
            tmp_filename.rename(output_filename)
            print(f'-> downloaded in {(timer() - start):.2f}s')
        else:
            print(f'No files to download for {year}')

        self.outputs['output_filenames'].write_text('\n'.join(all_filenames) + '\n')


if __name__ == '__main__':
    downloader.finalize()
    downloader.task_ctrl.print_status()
