import csv
import operator

def getKey(item):
  x = float(item[1])
  print x
  return x

ifile  = open('../data/modeBalanced/typicality_file_1.csv', "rb")
reader = csv.reader(ifile)
ofile  = open('../data/modeBalanced/typicality_file_1_sorted.csv', "wb")
writer = csv.writer(ofile, delimiter='\t')
 
rows = []
typ = []
reader.next()

for row in reader:
  typ.append(row[0])
  rows.append(row)

rows = sorted(rows, key=getKey)

typ = set(typ)

print("typ: ", len(typ))

for row in rows:
  writer.writerow(row)
 
ifile.close()
ofile.close()
