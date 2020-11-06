/* SimpleApp.scala */
import System.{exit, nanoTime}

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import hashing.Hasher
import hashing.MinHasher

object Main {

  def time[R](block: => R): R = {
      val t0 = System.nanoTime()
      val result = block    // call-by-name
      val t1 = System.nanoTime()
      println("Time: " + (t1 - t0) / 1000 + "ms")
      result
  }

  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("SimpleApplication")
      .config("spark.master", "local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    // val docs = Seq(
    //   "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000903.txt",
    //   "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000907.txt",
    //   "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000915.txt",
    //   "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000921.txt",
    //   "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000925.txt",
    // )
    // var df = spark.read.option("wholetext", true).text(docs: _*).toDF("text")
    val docs = Seq("adsdknc", "fkjdnfjfv")
    var df = docs.toDF("text")
    df = df.withColumn("id", monotonically_increasing_id())

    // Shingler
    val shingle_len = 5

    // Minhasher
    val minhash_len = 100
    val shingle_bins = 999999999
    val hashes = List.tabulate(minhash_len)(n => new Hasher(n, shingle_bins))
    val minhasher = new MinHasher(hashes)

    // Compute Hashed Shingles
    df = df.withColumn("tmp", array_repeat($"text", length($"text") - shingle_len + 1))
    val shingles_expr = "transform(tmp,(x,i) -> substring(x from i+1 for " + shingle_len + "))"
    df = df.withColumn("shingles", expr(shingles_expr))
    df = df.withColumn("hashed_shingles", array_distinct(transform($"shingles", (x)=> hash(x))))

    // Compute Minhashes
    val minhasherUDF = udf[Seq[Int], Seq[Int]](minhasher.getMinHashes)
    df = df.withColumn("minhashes", minhasherUDF('hashed_shingles))

    // Cross all combinations
    df = df.crossJoin(
      df.withColumnRenamed("hashed_shingles", "hashed_shingles2")
        .withColumnRenamed("id", "id2")
        .withColumnRenamed("minhashes", "minhashes2")
    )

    // Real Distance
    df = df.withColumn("JaccardSim",
                   size(array_intersect($"hashed_shingles", $"hashed_shingles2"))/
                   size(array_union($"hashed_shingles", $"hashed_shingles2")))

    // Approx distance
    val compareSignaturesUDF = udf(
      (s1: Seq[Int], s2: Seq[Int]) => s1.zip(s2).count({case (x,y) => x == y}).toFloat / s1.size.toFloat
    )
    df = df.withColumn("approxJaccardSim", compareSignaturesUDF($"minhashes", $"minhashes2"))
    
    df.select("id", "id2", "jaccardSim", "approxJaccardSim").show()
    spark.stop()
  }
}