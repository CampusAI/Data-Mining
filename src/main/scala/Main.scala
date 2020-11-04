/* SimpleApp.scala */
import System.nanoTime

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{monotonically_increasing_id, udf}
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
      "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000903.txt",
      "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000907.txt",
      "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000915.txt",
      "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000921.txt",
      "/home/fedetask/Downloads/text_dataset/Part1/awards_1990/awd_1990_00/a9000925.txt",
    )
    var df = spark.read.option("wholetext", true).text(docs: _*).toDF("text")
    df = df.withColumn("id", monotonically_increasing_id())


    // Shingler
    val shingle_len = 5
    val shingle_bins = 999999999
    val shingler = new Shingler(shingle_len, shingle_bins)

    // Minhasher
    val minhash_len = 100
    val hashes = List.tabulate(minhash_len)(n => new Hasher(n, shingle_bins))
    val minhasher = new MinHasher(hashes)

    // Compute Hashed Shingles
    val hashShinglesUDF = udf[Seq[Int], String](shingler.getHashShingles)

    df = df.withColumn("hashed_shingles", hashShinglesUDF('text))

    df = df.withColumn("id", monotonically_increasing_id())

    // Compute Minhashes
    val minhasherUDF = udf[Seq[Int], Seq[Int]](minhasher.getMinHashes)
    df = df.withColumn("minhashes", minhasherUDF('hashed_shingles))

    // Cross all combinations
    df = df.crossJoin(
      df.withColumnRenamed("hashed_shingles", "hashed_shingles2")
        .withColumnRenamed("id", "id_j")
        .withColumnRenamed("minhashes", "minhashes2")
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