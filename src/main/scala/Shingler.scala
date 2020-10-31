class Shingler(shingle_len: Int, hash_bins : Int) {
  def getShingles(doc : String) : Set[String] = {
    return doc.grouped(shingle_len).toSet
  }
  
  def getHash(str : String) : Int = {
    return math.abs(str.hashCode % hash_bins)
  }
  
  def getHashedShingles(doc : String) : scala.collection.SortedSet[Int] = {
    return collection.immutable.SortedSet[Int]() ++ getShingles(doc).map(getHash)
  }
}
