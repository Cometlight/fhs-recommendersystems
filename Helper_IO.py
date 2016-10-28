import csv

def read_file(fn):
    items = []
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        reader.next()                     # in case we have a header
        for row in reader:
            items.append(row[0])
    return items