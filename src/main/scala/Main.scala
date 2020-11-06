/* SimpleApp.scala */
import System.{exit, nanoTime}

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.functions.{array_repeat, arrays_zip, col, count, desc, exists, expr, hash, length, monotonically_increasing_id, transform, udf}
import shingler.Shingler
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

    val docs = Seq(
      "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/*",
    )
    var df = spark.read.option("wholetext", true)
      .text(docs: _*).toDF("text")
    df = df.withColumn("id", monotonically_increasing_id())


    // Shingler
    val shingle_len = 5
    val shingle_bins = 999999999
    // Minhasher
    val minhash_len = 100
    val hashes = List.tabulate(minhash_len)(n => new Hasher(n, shingle_bins))
    val minhasher = new MinHasher(hashes)

    // Compute Hashed Shingles

    df = df.withColumn("tmp", array_repeat($"text", length($"text") - shingle_len + 1))
    val shingles_expr = "transform(tmp,(x,i) -> substring(x from i+1 for " + shingle_len + "))"
    df = df.withColumn("shingles", expr(shingles_expr))
    df = df.withColumn("hashed_shingles", transform($"shingles", (x)=> hash(x)))

    df = df.withColumn("id", monotonically_increasing_id())

    // Compute Minhashes
    val minhasherUDF = udf[Seq[Int], Seq[Int]](minhasher.getMinHashes)
    df = df.withColumn("minhashes", minhasherUDF('hashed_shingles))

    // Cross all combinations
    df = df.crossJoin(
      df.withColumnRenamed("hashed_shingles", "hashed_shingles2")
        .withColumnRenamed("id", "id_j")
        .withColumnRenamed("minhashes", "minhashes2")
    ).filter($"id" < $"id_j")

    df = df.withColumn(
      "lsh_match", exists(
        arrays_zip(col("minhashes"), col("minhashes2")), x => x("minhashes") === x("minhashes2")
      )
    )

    // Real Distance
    val jaccardSimUDF = udf(
      (s1: Seq[Int], s2: Seq[Int]) => (s1.intersect(s2).toSet.size.toFloat )/ (s1 ++ s2).toSet.size.toFloat
    )
    println("Real Jaccard distance:")
    df = time{ df.withColumn("jaccardSim", jaccardSimUDF($"hashed_shingles", $"hashed_shingles2")) }

    // Aprox distance
    val compareSignaturesUDF = udf(
      (s1: Seq[Int], s2: Seq[Int]) => s1.zip(s2).count({case (x,y) => x == y}).toFloat / s1.size.toFloat
    )

    println("Comparing signatures:")
    df = time{ df.withColumn("approxJaccardSim", compareSignaturesUDF($"minhashes", $"minhashes2"))}
    df.select("id", "id_j", "jaccardSim", "approxJaccardSim").show()
    spark.stop()
  }
}