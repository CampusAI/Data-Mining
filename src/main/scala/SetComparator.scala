class SetComparator {
  def getJaccardSim(s1 : scala.collection.SortedSet[Int], s2 : scala.collection.SortedSet[Int]) : Float = {
    val intersection = s1.intersect(s2)
    val union = s1.union(s2)
    return intersection.size.toFloat/union.size.toFloat
  }

   def getJaccardSim(s1 : Set[Int], s2 : Set[Int]) : Float = {
    val intersection = s1.intersect(s2)
    val union = s1.union(s2)
    return intersection.size.toFloat/union.size.toFloat
  }
}