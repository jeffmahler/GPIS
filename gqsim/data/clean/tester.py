import csv
import glob
import os

def get_status(valid=('crash', 'ghost', 'good', '?')):
    response = None
    while response not in valid:
        response = input('status> ')
    return response

def main(pattern='*/*.world.xml'):
    results = []
    for fname in glob.glob(pattern):
        command = '../../gqsim -f {}'.format(fname)
        os.system(command)

        results.append((fname, get_status()))

    with open('results.csv', 'w') as f:
        writer = csv.writer(f)
        for row in results:
            writer.writerow(row)

if __name__ == '__main__':
    main()
