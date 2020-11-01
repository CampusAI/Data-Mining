package shingler

/** Bundles Shingler finding and hashing functionalities
  *
  * @param shingle_len the length (e.g. in characters) in which to split the document
  * @param hash_bins the number of bins into which to map the shingle hashes
  */
class Shingler(shingle_len: Int, hash_bins : Int) {

  /** Split document into shingles
    * @param doc Document which needs to be split
    * @return Set of shingles found in document
    */
  def getShingles(doc : String) : Set[String] = {
    return doc.grouped(shingle_len).toSet
  }
  
  /** Get hash of a string from 0 to hash_bins
    *
    * @param str String to get the hash from
    * @return Hash value
    */
  def getHash(str : String) : Int = {
    return math.abs(str.hashCode % hash_bins)  // TODO: Should we use our hasher?
  }
  
  /** Get a Set of hashed shingles of a document
    *
    * @param doc Document from which to get the hashed shingles
    * @return Set of hashes
    */
  def getHashShingles(doc : String) : Set[Int] = {
    return getShingles(doc).map(getHash)
  }
  
  /** Get a sorted Set of hashed shingles of a document
    *
    * @param doc Document from which to get the hashed shingles
    * @return sorted Set of hashes
    */
  def getOrderedHashShingles(doc : String) : scala.collection.SortedSet[Int] = {
    return collection.immutable.SortedSet[Int]() ++ getShingles(doc).map(getHash)
  }
}

// // Usage example:
// var doc = "abcde"
// var shingler = new Shingler(2, 3)
// var shingles = shingler.getOrderedHashShingles(doc1)
// shingles.foreach(element => println(element))