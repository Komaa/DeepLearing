import sys


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    :param iteration:   - Required  : current iteration (Int)
    :param total:       - Required  : total iterations (Int)
    :param prefix:      - Optional  : prefix string (Str)
    :param suffix:      - Optional  : suffix string (Str)
    :param decimals:    - Optional  : positive number of decimals in percent complete (Int)
    :param barLength:   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
