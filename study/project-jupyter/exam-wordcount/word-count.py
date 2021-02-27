# import SparkConf, SparkContext
from pyspark import SparkConf, SparkContext

# Set master node as the local machine. We call this application "WordCount"
conf = SparkConf().setMaster("local").setAppName("WordCount")
# Assign Spack context object to sc
sc = SparkContext.getOrCreate(conf)

# ---------------------- Implement your codes ----------------------------
# read data from Book.txt
# splits up each line of text 
# count how many times each unique values occur

# -------------------------------------------------------------------------

# print results
for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if (cleanWord):
        print(cleanWord.decode() + " " + str(count))
