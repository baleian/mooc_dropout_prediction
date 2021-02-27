# efficient Implementation

from pyspark import SparkContext, SparkConf
from time import time
conf = SparkConf().setMaster('yarn')
sc = SparkContext.getOrCreate(conf)

iters = 10
url_count = 6012
partitions = 4

def computeContribs(dsts, rank):
    num_dsts = len(dsts)
    contribs = []
    for dst in dsts:
        contribs.append((dst, rank/num_dsts))
    return contribs
"""
if len(sys.argv) != 2:
    print("Usage: pagerank <file> <output_file>")
    exit(-1)
"""
t0 = time()
data = sc.textFile("gs://dataproc-a707ebd5-2992-4b67-b43f-2f297b8ad78b-asia/notebooks/dgraph.txt", partitions).map(lambda line: line.split()).map(lambda words: (int(words[0]), int(words[1]))) 
links = data.distinct().groupByKey().partitionBy(partitions)
ranks = links.mapValues(lambda edge: 1.0)

for i in range(iters):
    contribs = links.join(ranks).flatMap(lambda src_dests_rank: \
                                         computeContribs(src_dests_rank[1][0], src_dests_rank[1][1]))
    ranks = contribs.reduceByKey(lambda x, y: x +y).mapValues(lambda rank: rank*0.85 + 1*0.15).partitionBy(partitions)

ranks.cache()
print( ranks.map(lambda x: x[1]).sum() )
print(ranks.sortBy(lambda x: -x[1]).collect())
print(time() - t0)
#ranks.saveAsTextFile("./output")