# IMporting SQL libraries
from pyspark.sql.types import *
# from pyspark.sql.functions import * 
from pyspark.sql.functions import sum as _sum

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

csv = spark.read.option("sep", "\t").csv('azureData3.csv', header=True, inferSchema=True)

csv.createTempView('mytable3')
csv.columns

spark.sql('select count(*) from mytable3').show()

# IMporting libraries
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="processed_text", outputCol="words", pattern="\\s+")

# stop words
add_stopwords = ["http","https","amp","rt","t","c","can", # standard stop words
     "#keithlamontscott","#charlotteprotest","#charlotteriots","#keithscott"] # keywords used to pull data)
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

# bag of words count
# Here we created bag of words as 10000 beecause more than that will take hours to compute and sometimes fails to.
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

# Creating pipeline of basic data cleaning and creating dataset with all features.
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])

# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(csv)
# Transform dataset with new pipelined features.
dataset = pipelineFit.transform(csv)

dataset.dtypes

dataset.show()

d = dataset.select('features')

d.show()

d.createOrReplaceTempView('mt')

a = d

# Splitting data for K-Means
(trainingData, testData) = a.randomSplit([0.75, 0.25], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

from pyspark.ml.clustering import KMeans

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
modelKM = kmeans.fit(trainingData)

# Make predictions
predictionsKM = modelKM.transform(testData)

wssse = modelKM.computeCost(predictionsKM)
print("Within Set Sum of Squared Errors = " + str(wssse))

centers = modelKM.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

predictionsKM.show()

# Splitting data for K-Means
(trainingData, testData) = a.randomSplit([0.75, 0.25], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

from pyspark.ml.clustering import BisectingKMeans

# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1)
modelBKM = bkm.fit(trainingData)

# Make predictions
predictions = modelBKM.transform(testData)

predictions.show()

wssse = modelBKM.computeCost(predictions)
print("Within Set Sum of Squared Errors = " + str(wssse))

dataset = dataset.select('processed_text', 'words', 'filtered', 'features')

dataset.createOrReplaceTempView('my')

# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1)
modelBKM = bkm.fit(dataset)
# Make predictions
predictions = modelBKM.transform(dataset)

dataset = predictions

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

trainingData.dtypes

from pyspark.sql.functions import *
trainingData = trainingData.select(col('prediction').alias('y'), col('features'))

from pyspark.ml.classification import LogisticRegression
# Build the model
lr = LogisticRegression(labelCol='y', maxIter=20, regParam=0.3, elasticNetParam=0, family = "binomial")

# Train model with Training Data
lrModel = lr.fit(trainingData)

testData = testData.select(col('prediction').alias('y'), col('features'))

trainingSummary = lrModel.summary

predictions = lrModel.transform(testData)

from pyspark.sql.functions import col
see = predictions.select('y', col('y').alias('label'), 'rawPrediction')

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print("Test: Area Under ROC: " + str(evaluator.evaluate(see, {evaluator.metricName: "areaUnderROC"})))

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

trainingData = trainingData.select(col('prediction').alias('y'), col('features'))

from pyspark.ml.classification import DecisionTreeClassifier

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="y", featuresCol="features", maxDepth=3)
# Only for cli
# dt = DecisionTreeClassifier(labelCol="y", featuresCol="features")
# Train model with Training Data
dtModel = dt.fit(trainingData)