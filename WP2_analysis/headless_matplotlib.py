import os
if os.getenv('HEADLESS'):
    import matplotlib
    matplotlib.use('agg')
