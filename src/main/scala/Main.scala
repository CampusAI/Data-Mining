/* SimpleApp.scala */
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.monotonically_increasing_id

import shingler.Shingler
import hashing.Hasher
import hashing.MinHasher
import comparator.Comparator

object Main {

  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("SimpleApplication")
      .config("spark.master", "local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    // Load data
    val docs = Seq((0, "qwertyuiopasdfghjkl"),
                   (1, "qwertyvkfjdfknfdjdfasdfg"))
    var df = docs.toDF("id", "text") // var or val?

    // Shingler
    val shingle_len = 3
    val shingle_bins = 100
    var shingler = new Shingler(shingle_len, shingle_bins)


    val hashShinglesUDF = udf[Seq[Int], String](shingler.getHashShingles)
    df = df.withColumn("hashed_shingles", hashShinglesUDF('text))
    df = df.crossJoin(df.withColumnRenamed("hashed_shingles", "hashed_shingles2").withColumnRenamed("id", "id_j"))

    val jaccardSimUDF = udf((set1: Seq[Int], set2: Seq[Int]) => (set1.intersect(set2).toSet.size.toFloat )/ (set1 ++ set2).toSet.size.toFloat)
    df = df.withColumn("jaccardSim", jaccardSimUDF($"hashed_shingles", $"hashed_shingles2"))
    df.select("id", "id_j", "jaccardSim").show()
    
    spark.stop()
  }
}