class SetComparator {
  def getJaccardSim(s1 : scala.collection.SortedSet[Int], s2 : scala.collection.SortedSet[Int]) : Float = {
    var intersection = s1.intersect(s2)
    var union = s1.union(s2)
    return intersection.size.toFloat/union.size.toFloat
  }
}