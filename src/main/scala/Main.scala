/* SimpleApp.scala */
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level

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

    // val logFile = "/home/oleguer/Documents/p6/Data-Mining/datasets/document.txt"
    // val logData = spark.read.textFile(logFile).cache()

    val docs = List("qwertyuiopasdfghjkl", "qwertyvkfjdfknfdjdfasdfg")

    // Shingler
    val shingle_len = 3
    val shingle_bins = 100
    var shingler = new Shingler(shingle_len, shingle_bins)

    // Minhasher
    val minhash_len = 20
    val hashes = List.tabulate(minhash_len)(n => new Hasher(n, shingle_bins))
    val minhasher = new MinHasher(hashes)

    // Comparator
    val comparator = new Comparator()

    // Mihashing pipeline (fast but approximate)
    val shingles = docs.map(shingler.getHashShingles)
    val minhashes = shingles.map(minhasher.getMinHashes)
    val approx_jaccard_dist = comparator.compareSignatures(minhashes(0), minhashes(1))
    println("Approx Jaccard distance: " + approx_jaccard_dist)

    // Real Jaccard distance (slow but accurate)
    val real_jaccard_dist = comparator.getJaccardSim(shingles(0), shingles(1))
    println("Real Jaccard distance: " + real_jaccard_dist)
    
    spark.stop()
  }
}