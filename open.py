#!/usr/bin/python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.sql import Row

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import NaiveBayesModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import DecisionTreeClassificationModel
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def parse_line(p):
    cols = p.split(' ')
    label = cols[0]
    if label not in ('Y', 'N'):
        return None

    label = 1.0 if label == 'Y' else 0.0
    fname = ' '.join(cols[1:])

    return Row(label=label, sentence=fname)


def train(spark):
    sc = spark.sparkContext
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=8000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    srcdf = sc.textFile('part.csv').map(parse_line)
    srcdf = srcdf.toDF()
    training, testing = srcdf.randomSplit([0.9, 0.1])

    wordsData = tokenizer.transform(training)
    featurizedData = hashingTF.transform(wordsData)
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    rescaledData.persist()

    trainDF = rescaledData.select("features", "label").rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()
    naivebayes = NaiveBayes()
    model = naivebayes.fit(trainDF)

    testWordsData = tokenizer.transform(testing)
    testFeaturizedData = hashingTF.transform(testWordsData)
    testIDFModel = idf.fit(testFeaturizedData)
    testRescaledData = testIDFModel.transform(testFeaturizedData)
    testRescaledData.persist()

    testDF = testRescaledData.select("features", "label").rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()
    predictions = model.transform(testDF)
    predictions.show()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("The accuracy on test-set is " + str(accuracy))
    model.save('Bayes20000')


def test(spark):
    sc = spark.sparkContext

    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=8000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    srcdf = sc.textFile('predict.csv').map(parse_line)
    testing = srcdf.toDF()

    model = DecisionTreeClassificationModel.load('Bayes20000')

    testWordsData = tokenizer.transform(testing)
    testFeaturizedData = hashingTF.transform(testWordsData)
    testIDFModel = idf.fit(testFeaturizedData)
    testRescaledData = testIDFModel.transform(testFeaturizedData)
    testRescaledData.persist()

    testDF = testRescaledData.select("features", "label").rdd.map(
        lambda x: Row(label=float(x['label']), features=Vectors.dense(x['features']))
    ).toDF()
    predictions = model.transform(testDF)
    predictions.select('prediction').write.csv(path='submit', header=True, sep=',', mode='overwrite')

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("The accuracy on test-set is " + str(accuracy))


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("Bigdata").getOrCreate()

    train(spark)

    spark.stop()
