filename ='ranking/data/amazon-electronics/amazon_electronic_datasets.csv'
total = sum(1 for line in open(filename))
print('行数有：', total)