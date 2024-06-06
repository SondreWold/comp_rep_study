# Meeting, 6th of June

*Who*: Philipp, Robert, Barbara, Sondre

## Motivation
- A central hypothesis in the up-and-coming field of mechanistic
  interpretability is the so-called "universality hypothesis": that
  different models learn similar representations for similar tasks.
- [Circuit Component Reuse Across Tasks in Transformer Language
  Models](https://openreview.net/forum?id=fpoAYV6Wsk), found some evidence
  supporting this claim through the use of activation patching on already
  trained GPT2 models between two similar tasks.
- [Break It Down: Evidence for Structural Compositionality in Neural
  Networks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/85069585133c4c168c865e65d72e9775-Abstract-Conference.html)
  also found some evidence for what they call "structural compositionality"
  in the Transformer: models break down complex tasks into subroutines that
  compute smaller units of the overall problem. This was done through
  weight masking.
- Will we observe similar results if we train Transformer models *from
  scratch* on datasets that are highly compositional by design? SCAN and
  PCFG SET contains some similar variable manipulation patterns that
  can be composed into compositional structures.

## Core idea
- Can we isolate circuits that solve specific functions when solving SCAN
  and PCFG SET?
- Within the same dataset, do circuits that approximate similar functions
  relate to each other? Is the circuit that can approximate the "remove
  first" operation in PCFG SET similar to the "remove second" operation? Do
  circuits for unary function transfer better to other unary functions
  compared to binary functions? etc...
- Does a circuit that approximate a variable manipulation pattern in SCAN
  transfer to the same pattern in PCFG SET?  And vice versa?
- Do the circuits compose according to how the data compose? Is the
  function "repeat" a repetition of "copy"?

## Contributions/directions
- We study the transferability of functions with respect to the specific variable manipulation
  pattern they perform, *without any noise*, both within the same task and across
  tasks.
- We perform an analysis on the compositionality of the identified circuits
  to see if there is a correlation between the representations of the model
  and the structure of the grammar used to generate the data.
- As we train models from scratch, compared to relying on pre-trained
  models, we study how model size influence circuit transferability
    - (My intuition is that smaller models are forced to create
      abstractions, compared to bigger models who can create input-output
      paths for almost all possible combinations)
- Similar to the "Break it Down" article, but in contrast to other related
  works in mechanistic interpretability, we rely on a circuit discovery
  method that is not dependent on the Transformer architecture, providing a
  more general framework for evaluation.
    - Since we do not rely on attention heads, it could be very interesting
      to do all the experiments with an RNN too and compare.


## Going forward:
We agreed upon using the next couple of weeks to synthesize and read up on
existing methods for finding subnetworks/circuits/masks/etc, and then take
it from there.
