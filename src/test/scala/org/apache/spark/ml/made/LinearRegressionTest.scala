package org.apache.spark.ml.made

import breeze.linalg.DenseVector
import com.google.common.io.Files
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  final val PREDICTION_COLUMN_NAME: String = "prediction"

  "Estimator" should "find correct weights" in {
    val model = getLinearRegression.fit(dataFrame)
    validateEstimator(model.getWeights)
  }

  "Model" should "predict result with high value of R2" in {
    val model = getLinearRegression.fit(dataFrame)
    validateModel(model, dataFrame)
  }

  "Estimator" should "work correctly after loading" in {
    val temporaryFolder = Files.createTempDir()
    new Pipeline()
      .setStages(Array(getLinearRegression))
      .write
      .overwrite()
      .save(temporaryFolder.getAbsolutePath)
    val model = Pipeline
      .load(temporaryFolder.getAbsolutePath)
      .fit(dataFrame)
      .stages(0)
      .asInstanceOf[LinearRegressionModel]
    validateEstimator(model.getWeights)
  }

  "Model" should "work correctly after loading" in {
    val temporaryFolder = Files.createTempDir()
    new Pipeline()
      .setStages(Array(getLinearRegression))
      .fit(dataFrame)
      .write
      .overwrite()
      .save(temporaryFolder.getAbsolutePath)
    val model = PipelineModel
      .load(temporaryFolder.getAbsolutePath)
      .stages(0)
      .asInstanceOf[LinearRegressionModel]
    validateModel(model, dataFrame)
  }

  private def getLinearRegression: LinearRegression = {
    new LinearRegression()
      .setFeaturesCol(FEATURES_COLUMN_NAME)
      .setLabelCol(TEST_COLUMN_NAME)
      .setPredictionCol(PREDICTION_COLUMN_NAME)
      .setNumberOfIterations(1000)
      .setAlpha(1.0)
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame): Unit = {
    val r2 = new RegressionEvaluator()
      .setLabelCol(TEST_COLUMN_NAME)
      .setPredictionCol(PREDICTION_COLUMN_NAME)
      .setMetricName("r2")
      .evaluate(model.transform(data))
    r2 should be > 0.99
  }

  private def validateEstimator(weights: DenseVector[Double]): Unit = {
    val delta = 0.01
    weights(0) should be(0.51 +- delta)
    weights(1) should be(42.0 +- delta)
    weights(2) should be(4.2 +- delta)
    weights(3) should be(-0.42 +- delta)
  }
}
