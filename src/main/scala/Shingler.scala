class Shingler(shingle_len: Int, hash_bins : Int) {
  def getShingles(doc : String) : Set[String] = {
    return doc.grouped(shingle_len).toSet
  }
  
  def getHash(str : String) : Int = {
    return math.abs(str.hashCode % hash_bins)
  }
  
  def getOrderedHashShingles(doc : String) : scala.collection.SortedSet[Int] = {
    return collection.immutable.SortedSet[Int]() ++ getShingles(doc).map(getHash)
  }
  
  def getHashShingles(doc : String) : Set[Int] = {
    return getShingles(doc).map(getHash)
  }
}

// // Usage example:
// var doc = "abcde"
// var shingler = new Shingler(2, 3)
// var shingles = shingler.getOrderedHashShingles(doc1)
// shingles.foreach(element => println(element))