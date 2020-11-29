package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

trait WithSpark {
  final val FIRST_COLUMN_NAME: String = "a"
  final val SECOND_COLUMN_NAME: String = "b"
  final val THIRD_COLUMN_NAME: String = "c"
  final val TEST_COLUMN_NAME: String = "result"
  final val FEATURES_COLUMN_NAME: String = "features"

  lazy val spark: SparkSession = WithSpark._spark
  lazy val sqlc: SQLContext = WithSpark._sqlc

  lazy val testDataSchema: StructType = new StructType()
    .add(FIRST_COLUMN_NAME, DoubleType)
    .add(SECOND_COLUMN_NAME, DoubleType)
    .add(THIRD_COLUMN_NAME, DoubleType)
    .add(TEST_COLUMN_NAME, DoubleType)

  lazy val testData: DataFrame = WithSpark._sqlc.read
    .option("header", "true")
    .schema(testDataSchema)
    // generated data
    .csv(getClass.getResource("/test.csv").getPath)

  lazy val dataFrame: DataFrame = new VectorAssembler()
    .setInputCols(Array(FIRST_COLUMN_NAME, SECOND_COLUMN_NAME, THIRD_COLUMN_NAME))
    .setOutputCol(FEATURES_COLUMN_NAME)
    .transform(testData)
    .drop(FIRST_COLUMN_NAME, SECOND_COLUMN_NAME, THIRD_COLUMN_NAME)
}

object WithSpark {
  lazy val _spark = SparkSession.builder
    .appName("Linear Regression")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc = _spark.sqlContext
}
