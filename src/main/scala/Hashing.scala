package hashing

import scala.util.Random

/** Represents an element of a Family of Hash function
 *
 *  @constructor define the hash function
 *  @param max_val the hash will be an integer from 0 to max_val
 *  @param seed the random hash parameters seed
 *  @param p prime number which should be larger than the largest inputed value
 */
class Hasher(seed: Int, max_val : Int, p : Int = 104729) {
  // https://stackoverflow.com/questions/19701052/how-many-hash-functions-are-required-in-a-minhash-algorithm
  private val random_generator = new scala.util.Random(seed)
  val a = 1 + 2*random_generator.nextInt((p-2)/2) // a odd in [1, p-1]
  val b = 1 + random_generator.nextInt(p - 2) // b in [1, p-1]
  
  /** Get the hash value of the inputed integer
   *  @param x the integer from which to get the hash value
   *  @return the hash value
   */
  def getHash(x : Int) : Int = {
    return ((a*x + b) % p) % max_val
  }
}

/** Implements set minhashing functionality
 *
 *  @constructor store the hashes
 *  @param hashes the list of hash functions to use (random permutations)
 */
class MinHasher(hashes : List[Hasher]) {

  /** Get the MinHash of a set applying the provided hash function (permutation)
    *  @param hasher the hash function used to permute the set ids
    *  @param set the ids from which to find the MinHash
    *  @return Minhash of the set once the permutation has been applied
    */
  def getMinHash(set : Set[Int])(hash : Hasher) : Int = {
    return set.map(hash.getHash).min
  }
  
  /** Get the MinHash List of the given set applying the stored hash functions (permutation)
    *  @param set the ids from which to find the MinHash List
    *  @return List of minhashes
    */
  def getMinHashes(set: Set[Int]) : List[Int] = {
    return hashes.map(getMinHash(set))
  }
}