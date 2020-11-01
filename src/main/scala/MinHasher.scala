import scala.util.Random

/** Represents an element of a Family of Hash function
 *
 *  @constructor define the hash function
 *  @param max_val the hash will be an integer from 0 to max_val
 *  @param seed the random hash parameters seed
 *  @param p prime number which should be larger than the largest inputed value
 */
class Hasher(max_val : Int, seed: Int, p : Int = 104729) {
  // https://stackoverflow.com/questions/19701052/how-many-hash-functions-are-required-in-a-minhash-algorithm
  private val random_generator = new scala.util.Random(seed)
  val a = 1 + 2*random_generator.nextInt((p-2)/2)// a odd in [1, p-1]
  val b = 1 + random_generator.nextInt(p - 2) // b in [1, p-1]
  
  /** Get the hash value of the inputed integer
   *  @param x Integer from which to get the hash value
   */
  def getHash(x : Int) : Int = {
    return ((a*x + b) % p) % max_val
  }
}