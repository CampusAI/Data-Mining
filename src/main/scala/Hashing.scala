package hashing

import scala.math.ceil
import scala.util.Random

/** Represents an element of a Family of Hash function
 *
 *  @constructor define the hash function
 *  @param max_val the hash will be an integer from 0 to max_val
 *  @param seed the random hash parameters seed
 *  @param p prime number which should be larger than the largest inputed value
 */
class Hasher(seed: Int, max_val : Int, p : Int = 104729) extends Serializable {
  // https://stackoverflow.com/questions/19701052/how-many-hash-functions-are-required-in-a-minhash-algorithm
  private val random_generator = new scala.util.Random(seed)
  val a = 1 + 2*random_generator.nextInt((p-2)/2) // a odd in [1, p-1]
  val b = 1 + random_generator.nextInt(p - 2) // b in [1, p-1]
  
  /** Get the hash value of the inputed integer
   *  @param x the integer from which to get the hash value
   *  @return the hash value
   */
  def getHash(x : Int) : Int = ((a*x + b) % p) % max_val
}

/** Implements set minhashing functionality
 *
 *  @constructor store the hashes
 *  @param hashes the list of hash functions to use (random permutations)
 */
class MinHasher(hashes : List[Hasher]) extends Serializable {

  /** Get the MinHash of a set applying the provided hash function (permutation)
    *  @param hasher the hash function used to permute the set ids
    *  @param set the ids from which to find the MinHash
    *  @return Minhash of the set once the permutation has been applied
    */
  def getMinHash(set : Seq[Int])(hasher : Hasher) : Int = set.map(hasher.getHash).min
  
  /** Get the MinHash List of the given set applying the stored hash functions (permutation)
    *  @param set the ids from which to find the MinHash List
    *  @return List of minhashes
    */
  def getMinHashes(set: Seq[Int]) : Seq[Int] = hashes.map(getMinHash(set))
}

/** Locally Sensitive Hashing class.
 *  Divide the minhash signatures in b bands of size r and hash each of the bands.
 *
 * @param b Number of bands in which the minhash signatures are divided.
 */
class LSH(b: Int) extends Serializable {

  /** Hash an integer array by accumulating the Int.hashCode function.
   *
   * @param array Sequence of integers.
   * @return Int hash.
   */
  def hashArray(array: Seq[Int]): Int = array.foldLeft(0)((tot, e) => (tot+e).hashCode())

  /** Perform LSH on the given minhash signature.
   *
   * @param signature Sequence of Int containing the minhash signature.
   * @return A sequence of hashes of length b.
   */
  def hashBands(signature: Seq[Int]): Seq[Int] = signature
    .grouped(ceil(signature.length.toDouble / b).toInt).toSeq
    .flatMap(band => Seq(hashArray(band)))
}