package org.apache.spark.ml.made

import breeze.linalg.{sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.util.MetadataUtils.getNumFeatures
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, PredictorParams}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{Dataset, Encoder}


trait LinearRegressionPredictorParams extends PredictorParams {
  final val numberOfIterations: IntParam = new IntParam(this, "numberOfIterations", "Number of iterations")
  final val alpha: DoubleParam = new DoubleParam(this, "alpha", "Learning rate")

  def setAlpha(value: Double): this.type = set(alpha, value)

  def setNumberOfIterations(value: Int): this.type = set(numberOfIterations, value)

  setDefault(alpha -> 1.0, numberOfIterations -> 100)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
    } else {
      SchemaUtils.appendColumn(schema, StructField(getPredictionCol, new VectorUDT()))
    }
    if (schema.fieldNames.contains($(labelCol))) {
      SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)
    } else {
      SchemaUtils.appendColumn(schema, StructField(getLabelCol, DoubleType))
    }
    schema
  }
}

class LinearRegression(override val uid: String)
  extends Estimator[LinearRegressionModel]
    with LinearRegressionPredictorParams with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("LinearRegression"))

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val numberOfFeatures = getNumFeatures(dataset, $(featuresCol))
    var weights: BreezeDenseVector[Double] = BreezeDenseVector.zeros(numberOfFeatures + 1)
    val featuresOutputColumnName: String = "features_output"
    val onesColumnName: String = "ones"

    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    val assembledFeatures: Dataset[Vector] = new VectorAssembler()
      .setInputCols(Array(onesColumnName, $(featuresCol), $(labelCol)))
      .setOutputCol(featuresOutputColumnName)
      .transform(dataset.withColumn(onesColumnName, lit(1)))
      .select(featuresOutputColumnName)
      .as[Vector]

    for (iteration <- 0 to $(numberOfIterations)) {
      val result = assembledFeatures.rdd.mapPartitions((data: Iterator[Vector]) => {
        val onlineSummarizer = new MultivariateOnlineSummarizer()
        data.foreach(value => {
          val x = value.asBreeze(0 until weights.size).toDenseVector
          val y = value.asBreeze(weights.size)
          val loss = sum(x * weights) - y
          onlineSummarizer.add(fromBreeze(x * loss))
        })
        Iterator(onlineSummarizer)
      }).reduce(_ merge _)

      weights -= $(alpha) * result.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights))).setParent(this)
  }
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel protected[made](override val uid: String, weights: Vector)
  extends RegressionModel[Vector, LinearRegressionModel]
    with LinearRegressionPredictorParams with MLWritable {
  def this(weights: Vector) = this(Identifiable.randomUID("LinearRegressionModel"), weights)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights))

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val params = Tuple1(weights.asInstanceOf[Vector])
      sqlContext.createDataFrame(Seq(params)).write.parquet(path + "/vectors")
    }
  }

  override def predict(features: Vector): Double = {
    val x = BreezeDenseVector.vertcat(BreezeDenseVector(1.0), features.asBreeze.toDenseVector)
    sum(x * weights.asBreeze.toDenseVector)
  }

  def getWeights: BreezeDenseVector[Double] = {
    weights.asBreeze.toDenseVector
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()
      val vectors = sqlContext.read.parquet(path + "/vectors")
      val params = vectors.select(vectors("_1").as[Vector]).first()

      val model = new LinearRegressionModel(params)
      metadata.getAndSetParams(model)
      model
    }
  }
}
