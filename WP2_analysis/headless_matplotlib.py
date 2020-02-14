import os
if os.getenv('HEADLESS').lower() == 'true':
    import matplotlib
    matplotlib.use('agg')
