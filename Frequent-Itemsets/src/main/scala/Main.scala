import System.{exit, nanoTime}
import scala.collection.mutable.WrappedArray
import org.apache.spark.sql.{Column, SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import spark.implicits._

object Main {
    val s = 0.5


  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Time: " + (t1 - t0).toFloat / 1000000000.0 + "s")
    result
  }

  def loadData(path: String) : DataFrame = {
    var data = spark.read.text(path)
      .toDF("baskets_str")
      .withColumn("baskets", split('baskets_str, " ").cast("array<int>"))
    data
  }

  def combo(a1: WrappedArray[Int], a2: WrappedArray[Int]): Array[Array[Int]] = {
    var a = a1.toSet
    var b = a2.toSet
    var res = a.diff(b).map(b+_) ++ b.diff(a).map(a+_)
    return res.map(_.toArray.sortWith(_ < _)).toArray
  }
  val comboUDF = udf[Array[Array[Int]], WrappedArray[Int], WrappedArray[Int]](combo)

  def getCombinations(df: DataFrame): DataFrame = {
    df.crossJoin(df.withColumnRenamed("itemsets", "itemsets_2"))
      .withColumn("combinations", comboUDF(col("itemsets"), col("itemsets_2")))
      .select("combinations")
      .withColumnRenamed("combinations", "itemsets")
      .withColumn("itemsets", explode(col("itemsets")))
      .dropDuplicates()
  }

  def countCombinations(data : DataFrame, combinations: DataFrame) : DataFrame = {
    data.crossJoin(combinations)
      .where(size(array_intersect('baskets, 'itemsets)) === size('itemsets))
      .groupBy("itemsets")
      .count
  }

  def loadFakeData() : DataFrame = {
    var data = Seq("1 ",
                  "1 2 ",
                  "1 2",
                  "3",
                  "1 2 3 ",
                  "1 2 ")
      .toDF("baskets_str")
      .withColumn("baskets", split('baskets_str, " ").cast("array<int>"))
      data
  }


  def main(): Unit = {
    // Start Spark
    val spark = SparkSession.builder.appName("FreqItemsets")
      .master("local[*]")
      .getOrCreate()

    // Load data
    // val path = "/home/oleguer/Documents/p6/Data-Mining/Frequent-Itmes/datasets/T10I4D100K.dat"
    // val data = loadData(path)
    val data = loadFakeData()
    val basket_count = data.count

    var itemset : DataFrame = data
                                .select(explode('baskets))
                                .na.drop
                                .dropDuplicates()
                                .withColumnRenamed("col", "itemsets")
                                .withColumn("itemsets", array('itemsets))
    var itemset_count : DataFrame = countCombinations(data, itemset).filter('count > s*basket_count)
    var itemset_counts = List(itemset_count)

    var stop = false
    while(!stop) {
      itemset = getCombinations(itemset_count.select("itemsets"))
      itemset_count = countCombinations(data, itemset).filter('count > s*basket_count)
      stop = (itemset_count.count == 0)
      if (!stop) {
        itemset_counts = itemset_counts :+ itemset_count
      }
    }
    println(itemset_counts.length)
    for (i <- itemset_counts) i.show()
  }
}






