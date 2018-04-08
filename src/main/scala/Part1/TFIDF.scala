package Part1

import java.util.regex.Pattern

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.hadoop.fs.FileSystem
import java.io._

import SparkUtils._
import org.apache.hadoop.fs.{FSDataOutputStream, Path}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object TFIDF {
  val regex = "[\\.\\,\\:\\-\\!\\?\\n\\t,\\%\\#\\*\\|\\=\\(\\)\\\"\\>\\<\\/]"
  val pattern = Pattern.compile(regex)

  def clean: String => String = pattern.matcher(_).replaceAll(" ").split("[ ]+").mkString(" ")

  def main(args: Array[String]) = {
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("TfIdfSpark")
      .set("spark.driver.memory", "3g")
      .set("spark.executor.memory", "2g")

    val sc = SparkSession.builder.config(conf).getOrCreate()
    import sc.implicits._

    val tweets = sc.read.format("csv").option("header", "true").load(args(0))
    val df2 = tweets.filter(tweets("text").isNotNull)
    val cleanUdf = udf(clean)
    val df = df2.withColumn("text",  cleanUdf($"text"))

    val splits = df.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    val hashingTF = new HashingTF()
      .setInputCol(remover.getOutputCol).setOutputCol("features").setNumFeatures(100)

    val indexer = new StringIndexer()
      .setInputCol("airline_sentiment")
      .setOutputCol("label")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    val nb = new NaiveBayes()

    val pipelineLR = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, indexer, lr))
    val pipelineNB = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, indexer, nb))

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val cvLR = new CrossValidator()
      .setEstimator(pipelineLR)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(6)

    val cvNB = new CrossValidator()
      .setEstimator(pipelineNB)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(6)

    val predLabelLR = cvLR.fit(training).transform(test).select("prediction","label").rdd.map{ case Row(prediction: Double, label: Double) => (prediction, label) }
    val predLabelNB = cvNB.fit(training).transform(test).select("prediction","label").rdd.map{ case Row(prediction: Double, label: Double) => (prediction, label) }

    val fs = FileSystem.get(sc.sparkContext.hadoopConfiguration)
    val path: Path = new Path(args(1))
    if (fs.exists(path)) {
      fs.delete(path, true)
    }
    val dataOutputStream: FSDataOutputStream = fs.create(path)
    val bw: BufferedWriter = new BufferedWriter(new OutputStreamWriter(dataOutputStream, "UTF-8"))
    bw.write(evaluateModel(predLabelLR, args(1), "Logistic Regression"))
    bw.write(evaluateModel(predLabelNB, args(1), "=== Naive Bayes ==="))
    bw.close
  }
}