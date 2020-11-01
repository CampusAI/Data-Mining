package comparator

/** Bundle of comparison functions
  */
class Comparator {

  /** Get Jaccard similarity between two sorted sets
   * @param s1 First sorted set
   * @param s2 Second sorted set
   * @return Jaccard similarity (s1, s2)
   */
  def getJaccardSim(s1 : scala.collection.SortedSet[Int], s2 : scala.collection.SortedSet[Int]) : Float = {
    val intersection = s1.intersect(s2)
    val union = s1.union(s2)
    return intersection.size.toFloat/union.size.toFloat
  }

  /** Get Jaccard similarity between two unordered sets
   * @param s1 First unordered set
   * @param s2 Second unordered set
   * @return Jaccard similarity (s1, s2)
   */
  def getJaccardSim(s1 : Set[Int], s2 : Set[Int]) : Float = {
    val intersection = s1.intersect(s2)
    val union = s1.union(s2)
    return intersection.size.toFloat/union.size.toFloat
  }

  /** Get proportion of matching signatures
   * @param l1 First list of signatures
   * @param l2 Second list of signatures
   * @return Matching signature proportion
   */
  def compareSignatures(l1 : List[Int], l2 : List[Int]) : Float = {
    assert(l1.size == l2.size) // Signature lengths should match
    return l1.zip(l2).count({case (x,y) => x == y}).toFloat / l1.size.toFloat
  }
}