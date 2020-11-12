import scala.collection.mutable.WrappedArray

object Main {

  // Functions to get L_{k+1} from L_k
  def combo(a1: WrappedArray[Int], a2: WrappedArray[Int]): Array[Array[Int]] = {
    var a = a1.toSet
    var b = a2.toSet
    var res = a.diff(b).map(b+_) ++ b.diff(a).map(a+_)
    return res.map(_.toArray).toArray
  }

  def main(args: Array[String]): Unit = {
    // Start Spark
    val spark = SparkSession.builder.appName("test")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    // Dataset
    var data = Seq(Set(1, 2, 3), Set(3, 4, 5), Set(6, 2, 4)).toDF("items")

    // Create combinations UDF
    val comboUDF = udf[Array[Array[Int]], WrappedArray[Int], WrappedArray[Int]](combo)

    // All possible pairs
    data = data.crossJoin(data.withColumnRenamed("items", "items_2"))
    // Create new column with all combinations
    data = data.withColumn("combinations", comboUDF(col("items"), col("items_2")))

    // Create a new dataframe with one combination per row
    data = data.select("combinations").withColumn("combinations", explode(col("combinations"))).dropDuplicates()

    data.show(false)

  }
}






