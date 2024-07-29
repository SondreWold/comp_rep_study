# Meeting, July 29.
Philipp and Sondre

## Immediate next steps
- Fix bug (Suspects: Intersection, IoMin/IoU, Complement). 
  - A intersection A => A (1 for IoU, 1 for IoMin, IntersectionFraction)
  - A union A => A
  - A intersection A' => empty set
  - A diff A => empty set
  - (A union B) intersection A => A
  - A union empty set => A
- Do some experiments with the subnetwork arithemtics (copy_echo, copy, echo).
  - Identify which layers to operate on (Intersection of all atomic circuits).
  - Do experiments where the "common layers" are left untouched.
  - Obtain 6x3 matrix.
 
## Looking into the future
- Compare pruning methods.
