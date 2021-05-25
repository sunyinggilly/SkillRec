# -*- coding: utf-8 -*-
from Utils.JobReader import read_jd
import sys
from pyspark import *
from pyspark.sql import *
from pyspark.mllib.fpm import FPGrowth
import pickle
reload(sys)
sys.setdefaultencoding('utf-8')

def connect_spark():
    conf = SparkConf()
    conf = conf.setAppName('SYem')
    sc = SparkContext(conf=conf)
    #sc.setLogLevel("ERROR")
    spark = SparkSession \
        .builder \
        .appName("SYem") \
        .config("spark.some.config.option", "some-value") \
        .enableHiveSupport() \
        .getOrCreate()
    return sc, spark


if __name__ == "__main__":
    sc, spark = connect_spark()
    print("read data")
    transactions = [x[0] for x in read_jd()]
    print("parallelize")
    rdd = sc.parallelize(transactions, 100)
    print("train")
    model = FPGrowth.train(rdd, minSupport=0.005, numPartitions=100)
    print("collect")
    result = model.freqItemsets().map(lambda x: (x.items, x.freq)).collect()
    print(len(result))
    with open("tmp/item_set.pkl", "wb") as f:
        pickle.dump(result, f)
