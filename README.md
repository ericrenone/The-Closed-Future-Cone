# The Closed Future Cone

*Compact Causal Geometry as the Native Substrate of Learned Representations.*

Two giants of early twentieth-century mathematics each made one move that the field has never stopped paying for. Hermann Minkowski looked at Einstein's 1905 relativity and said: this is geometry, not algebra — time is not an external parameter, time and space are coordinates on a single four-dimensional manifold with signature $(-,+,+,+)$, and the structure that governs causation is the light cone. L. E. J. Brouwer, working in topology a few years later, proved that every continuous self-map of a compact convex set into itself has a fixed point — the most famous purely existential result in the discipline, and the one Brouwer himself would later argue should never have been believed without a constructive proof.

These two ideas — *time as a geometric object* and *the fixed point as the inevitable consequence of compactness + continuity* — have not been brought together to describe learned representations in deep networks. They should be. This document is the proposal for a single geometric object that lives at their intersection: the **Closed Future Cone**, the compact causal substrate on which a forward pass *is* a Brouwer iteration, and on which the model's representation of *time itself* takes the form not of an unbounded direction but of a closed curve — a loop, a ring, a circle the network walks around.

The synthesis is new. The pieces are not.

---

## 1. A Thought Experiment to Begin

A neural network is asked the question: *What day of the week is three days after Friday?* A child can answer this; so can a calculator with a modulo operator; so, now, can a language model with eight billion parameters. The question is not whether the model can answer. The question is *how it represents the question internally* — what shape, in its activation space, the concept "day of the week" takes.

The naïve guess, the one the field believed for years, is that the model holds a list. Monday, Tuesday, Wednesday, …, Sunday. Each day is a vector in some high-dimensional space; each vector points in roughly its own direction; the answer is found by some computation over these seven directions. This is the *linear representation hypothesis*, and it has a long and honourable history.

It is also wrong, in a specific and beautiful way. When Engels, Liao, Michaud, Gurnee and Tegmark (*Not All Language Model Features Are Linear*, NeurIPS 2024, arXiv:2405.14860) trained sparse autoencoders on GPT-2 and Mistral-7B and went looking for what the days-of-the-week feature actually looked like, they found a **circle**. Seven points, arranged in a two-dimensional plane in the model's residual stream, equally spaced on a ring, in the right cyclic order — Monday next to Tuesday next to Wednesday, and Sunday next to Monday closing the loop. When the model is asked the *three days after Friday* question, the operation it performs internally is not a database lookup. The model rotates around the ring. The arithmetic of weekday addition is, mechanistically, *rotation by an angle*.

The same paper finds the same picture for months of the year (a twelve-pointed ring), and Karkada, Olah, Tegmark and collaborators (*Symmetry in language statistics shapes the geometry of model representations*, arXiv:2602.15029, February 2026) — the 2026 paper that supplies the analytical proof for why this happens — show that the rings are not accidents. The geometry is *forced* by the co-occurrence symmetry of the underlying language: whenever a concept's tokens admit a finite group of symmetries (the cyclic group $\mathbb{Z}_7$ for weekdays, $\mathbb{Z}_{12}$ for months), the optimal representational manifold is its corresponding group manifold. Cyclic symmetry implies a circle. The proof goes through whether one trains the model or not.

Consider what this means. The internal representation of the most universally cyclic of human concepts — *time itself* — is a **closed timelike curve**. Not a direction. Not an axis. A loop. The model's "future" returns to its "past," and the topology of its representation enforces this exactly.

This is the seed observation around which everything in this document is organised.

---

## 2. Why This Object Wants to Be Minkowskian

Minkowski's insight in 1908 was that time and space, separately, are projections — "shadows," in his word — of a single four-dimensional object with one timelike direction and three spacelike directions, distinguished by a metric signature rather than by ontology. The structure of causation in such a space is determined entirely by the *light cone*: at each event, the set of points reachable by signals of finite speed forms a cone with a definite interior (timelike — causally connected), boundary (lightlike — at the limit of causal contact), and exterior (spacelike — outside causation).

Now consider what a forward pass through a deep network is, geometrically.

At each layer, every token's residual stream vector is updated by some combination of contributions from a finite set of other tokens — specifically those it attends to in the attention layer, plus its own previous content via the residual connection. The set of token positions that can influence token $i$ at layer $\ell$ is determined by which positions $i$ attends to over the $\ell$ layers up to that point. This is not a metaphor for a light cone. **It is a light cone.** Information at token $j$ at the input cannot reach token $i$ at layer $\ell$ unless there exists a sequence of attention edges connecting them through the layer stack. The causal structure of information flow in a transformer is *exactly* the light-cone structure of a discrete Minkowski space whose time coordinate is layer depth.

In a causal (decoder-only) language model, this cone is open in one direction (future tokens cannot influence past tokens) and closed in the other. In a bidirectional model, it is closed in both. Either way, the light-cone structure of the architecture is a structural fact, not an emergent property.

Pair this with the Engels et al. observation. Time as a concept, *inside* the representation, is a circle. Time as the structure of *computation*, across layers, is a light cone. The model lives inside the latter and represents the former.

This is the geometry the proposal is named for. The light cone of the computation, and the closed timelike curve of the time concept it represents, together constitute the **Closed Future Cone** — a compact, causally structured object on which the network operates.

---

## 3. Why Brouwer

A continuous self-map $f : K \to K$ of a compact convex set $K$ has a fixed point: a point $x^* \in K$ with $f(x^*) = x^*$. This is the Brouwer fixed-point theorem, proved in 1911, and it is the prototype of a *purely existential* mathematical statement. The theorem asserts that $x^*$ exists but offers no instruction for finding it. You know it is there; you do not know where.

Brouwer himself, by the 1920s, had become the founder and chief advocate of *intuitionism*: the position that mathematical objects must be *constructed*, that the law of the excluded middle is not universally valid, that "there exists" must mean "we can build." The Brouwer–Hilbert controversy of the 1920s — *Grundlagenstreit*, the foundational struggle — pitted Brouwer's constructive demand against David Hilbert's formalism, in which existence proofs by contradiction or by appeal to compactness were perfectly admissible. Hilbert won the institutional fight; classical mathematics today is formalist, not intuitionist. Brouwer's own fixed-point theorem, the result for which he is best known to the wider world, is the canonical example of a result he later refused to accept *as he himself had originally proved it*, because the proof was non-constructive.

Consider what is happening when a transformer runs a forward pass on a long sequence of tokens. Each block performs an update on the residual stream: $h \leftarrow h + \text{Attn}(h) + \text{FFN}(h)$. After many such blocks, what does the limiting behaviour look like? When researchers ask whether a transformer can *implement* an iterative algorithm — gradient descent in-context, Newton's method for in-context regression, fixed-point iteration of a learned operator — they are not asking whether transformers can approximate functions in general. They are asking whether the forward pass *is* a constructive fixed-point iteration.

Hu, Liu, Chen, Wu and Liu (*Universal Approximation with Softmax Attention*, arXiv:2504.15956, updated December 2025) prove that two-layer self-attention is a universal approximator of continuous sequence-to-sequence functions on compact domains and, separately, that attention-only stacks are capable of approximating gradient descent in-context. Li, Jiao, Huang, Wei and Chen (arXiv:2506.05200, 2025) establish that in-context learning in transformers can be understood as a universal approximation result over the space of functions, with the transformer functioning as an algorithm approximator. The pattern that emerges across this 2024–2026 line of work is one in which the transformer is not merely a function but an *iterated operator* — and the value it converges to over depth is, structurally, a fixed point.

The compactness condition Brouwer needs is exactly the compactness condition the field has been quietly assuming for years. Sparse autoencoder atoms live in the convex hull of activations (Fel et al., *Into the Rabbit Hull*, arXiv:2510.08638, 2025). Attention outputs live in the convex hull of value vectors. The residual stream, under common normalization conventions, lives on a bounded sphere or in a bounded ball. The model is not operating on an unbounded Euclidean space; it is operating on a compact convex object, and its forward pass is a continuous self-map of that object.

By Brouwer, a fixed point exists.

By the constructive achievement of the transformer, *the iteration finds it*.

The Brouwer–Hilbert controversy, a century late, has a verdict the philosophers of mathematics did not anticipate. Hilbert's formalism gave us the theorem. Brouwer's intuitionism gave us the demand that the theorem be constructive. **The transformer is the constructive realization of Brouwer's non-constructive theorem.** Each forward pass walks the fixed point into being. The model does not merely *know* that an answer exists; it *constructs* it, layer by layer, through an iteration that is provably contractive on the compact convex hull where representations actually live.

The synthesis says: the geometric object on which deep networks compute is compact (so a fixed point exists by Brouwer), it is causally structured (so the iteration respects a light cone in the sense of Minkowski), and the closed loops the model uses to represent cyclic concepts like time are exactly the topologically non-trivial obstructions to that fixed point being a single point — they are the *attractors* of the forward pass, not the *termini*.

---

## 4. A Second Thought Experiment

Consider an ant walking on a clock face. The clock has twelve hour-marks arranged on a circle. Three positions around — that is the question the ant is asked: *given that I am at noon, where am I three hours later?* The ant does not consult a table. The ant walks, knowing only the local direction "forward along the rim." It arrives, after three steps, at the three-position. The ant's *computation* is a translation along the circle; the circle's *topology* is what makes the answer well-defined modulo twelve; and at no point does the ant ever need to leave the rim.

A transformer doing modular weekday arithmetic is the ant. The ring is the representational manifold the model has learned. The rotation is the operation the residual stream performs. The fact that the rotation closes — that seven steps along the weekday ring returns to where it started — is the topological consequence of the circle.

Now imagine the same ant, but now living on a clock face that has been *fibered over time*: at each instant, the ant lives on its current clock-face, and the clock-faces stack vertically, one for each successive layer of the model. The ant cannot reach upward through the stack arbitrarily — it can only reach upward in the directions determined by the attention pattern at each level, which forms a *cone* opening into the future. The ant's accessible neighborhood at layer $\ell$ from a starting position at layer $0$ is the intersection of (a) the original ring (cyclic structure of the concept), and (b) the layered attention cone (causal structure of the computation).

This intersection — the ring inside the cone — is the Closed Future Cone. A finite-dimensional, compact, causally structured object on which the model walks around.

The point of the thought experiment is that two distinct facts about deep networks have been hiding the same geometric object. The "concept manifold" literature finds rings. The "circuit analysis" literature finds light cones. Both are correct. The object they describe — *together* — is what the field has been groping toward without naming.

---

## 5. The Geometry Made Precise

The Closed Future Cone (CFC) is, in the proposal's strict form, the structure consisting of:

A **compact base manifold** $\mathcal{M}$, the representational geometry the model uses. For concepts admitting cyclic group symmetry, $\mathcal{M}$ is locally a torus $T^k$ (Karkada et al. 2026 prove this analytically for $\mathbb{Z}_n$ symmetries; the days-of-week circle is the $k=1$ case). For concepts with continuous translation symmetry like position in text, $\mathcal{M}$ contains the real line locally but compactifies globally via positional encoding.

A **layer-time foliation**, indexing the layers $\ell = 0, 1, \ldots, L$ of the network as discrete time slices. At each slice, $\mathcal{M}$ is the configuration available to the residual stream.

A **causal structure** on $\mathcal{M} \times \{0, \ldots, L\}$ given by the attention pattern. From any point $(x, \ell)$, the *future cone* is the set of points $(y, \ell')$ with $\ell' > \ell$ such that there exists an attention path from $x$'s token at $\ell$ to $y$'s token at $\ell'$. This is the discrete analogue of the open future light cone of an event in Minkowski space.

A **fixed-point operator** $F$ on $\mathcal{M}$, induced by the iterated application of attention blocks. By Brouwer (compact convex domain + continuous map), $F$ has at least one fixed point in the convex hull of $\mathcal{M}$.

The non-trivial topology of $\mathcal{M}$ — the fact that, for cyclic concepts, $\mathcal{M}$ is a circle and not a disc — is what prevents Brouwer's fixed point from being globally attractive. The fundamental group $\pi_1(\mathcal{M}) = \mathbb{Z}$ for a circle means there are *winding-number* classes of trajectories: a forward pass that walks around the ring once is topologically distinct from one that walks around twice. This is exactly what the model uses to compute modular arithmetic. The "answer" to *three days after Friday* is not a point on the ring; it is the *image* of a specific path with specific winding number applied to the starting point.

In Minkowski's spacetime, closed timelike curves are pathological — they enable causal paradoxes. In the compact base $\mathcal{M}$ of a representational space, closed timelike curves are the *whole point*. They are how the model represents time without representing infinity. The model has no Sunday-after-eternity; it has only the next position on the seven-cycle. Compactness saves it from the paradox by removing the need for one.

---

## 6. Why This Is Not Just Linear Representation, And Not Just Polytopal Representation

The Linear Representation Hypothesis (Park, Choe & Veitch, ICML 2024) says concepts are *directions*. Directions live in unbounded Euclidean space; they have no topology beyond local linear structure. A direction is the projection of the CFC onto a single tangent vector at a single point. It is what you see if you look at a small piece of $\mathcal{M}$ and forget that it closes.

The Minkowski Representation Hypothesis (Fel et al., 2025) says concepts are *convex polytopes*, assembled by multi-head attention as a Minkowski sum of head-wise value sets. Polytopes are *compact convex objects*. They have a topology — but it is the trivial topology of a disc. They do not have winding numbers, they do not have closed timelike curves, they cannot natively represent cyclic time.

The Closed Future Cone hypothesis says concepts live on a compact base manifold $\mathcal{M}$ whose topology is determined by the symmetry group of the underlying co-occurrence statistics (Karkada et al., 2026), foliated by layer-time, with a causal cone structure determined by attention. **LRH is the local linearization. MRH is the genus-zero (disc) case. CFC is the general case in which $\mathcal{M}$ may have non-trivial fundamental group.**

The Engels et al. circle is the simplest non-trivial case: a one-dimensional compact manifold with $\pi_1 = \mathbb{Z}$. The Karkada et al. proof shows that whenever the language exhibits a finite cyclic symmetry, this is the geometry forced. More elaborate symmetries should produce richer manifolds. The 2-torus is what one would expect for two-cycle composite concepts (day × month); $S^2$ would be expected for spherically symmetric concepts; products of circles for vector-cyclic concepts. The Engels paper is the data point. Karkada et al. is the theorem. The CFC is the framework that organizes them.

---

## 7. A Third Thought Experiment

Consider what it means for a recurrent neural network to *measure time*.

Jarne, Yoon, Eissa, Kilpatrick and Josić (*Task-Parametrized Dynamics: Representation of Time and Decisions in Recurrent Neural Networks*, bioRxiv:2025.09.15.676356, September 2025) trained RNNs on delayed decision-making tasks of increasing complexity and analyzed what computational strategies emerge. The networks settled, across runs, into one of two regimes for representing elapsed time: *oscillatory dynamics* (a closed cycle in state space, with elapsed time encoded by phase) or *integration dynamics* (a ramping trajectory, with elapsed time encoded by amplitude). The same task, different initializations, two qualitatively distinct geometries.

Notice what those two geometries are. The first is a *closed timelike curve* — the activity loops around in state space, and "time" is the phase angle of that loop. The second is an *open* trajectory — activity ramps along a direction, and "time" is the distance traveled. The first is what the network *prefers* for tasks with cyclic structure (rhythmic timing, modular decisions). The second is what the network *prefers* for tasks with linear structure (evidence accumulation, deadline approach).

The two regimes are not in competition. They are the two ways a network can represent time on a compact manifold $\mathcal{M}$: either the trajectory closes (the manifold is a circle or torus) or it does not (the manifold is an interval, locally a line). The choice between them is determined by the task; the geometry follows.

This is the CFC picture at the population level. Engels et al. find the *static* signature of the circle in language model activations. Jarne et al. find the *dynamic* signature in trained RNNs. They are looking at the same object from two angles. One is the spatial cross-section; the other is the temporal trajectory along the layer-time foliation. Both find that representational time is a closed loop wherever the data's underlying symmetry says it should be.

---

## 8. The Forward Pass as Compactness Argument

Here is what changes when one takes the CFC seriously as an organizing object rather than as a metaphor.

When a transformer is given a long prompt, the forward pass propagates information from the input through $L$ blocks. Each block is a continuous map. The composition of $L$ continuous maps on a compact convex domain is a continuous self-map of a compact convex domain. By Brouwer, this map has a fixed point. By the Banach fixed-point theorem (a constructive strengthening that requires a contraction property), if the map is contractive in some metric, the iteration converges to that fixed point from any starting condition. Recent work on **implicit transformers** and **deep equilibrium models** (Bai, Kolter & Koltun, NeurIPS 2019; the line that runs through Anil et al., 2022, and the universal-approximation-via-Leray-Schauder result of arXiv:2409.00841) makes this connection completely explicit: a sufficiently deep transformer is operationally equivalent to a network with a *single block iterated to convergence*. The forward pass *is* a fixed-point iteration. The depth is just the number of iterations one has spent.

What does this mean for interpretability? It means the "answer" the model gives to a question — the residual stream content at the final layer, at the final token — is not a quantity assembled piece by piece. It is the fixed point of an operator the model has learned, evaluated *constructively* by iteration. The model knows what the answer is by walking to it. The deeper the model, the more iterations it gets to walk.

What does it mean for the structure of representations? It means the **non-trivial topology of the representational manifold $\mathcal{M}$ — the closed loops, the rings, the cycles — corresponds to the equivalence classes of trajectories that converge to the same fixed point**. Two prompts that differ by a full cycle around the days-of-week ring (e.g., "what day is three days after Friday" and "what day is ten days after Friday") yield the same final answer because they correspond to trajectories with different winding numbers around the same fixed point of the iteration. The model's *invariances* are the topology of its representational manifold.

What does it mean for Brouwer's philosophical commitment? It vindicates intuitionism in a way Brouwer himself could not have anticipated. The fixed point is not just guaranteed to exist; in the trained network, it is *constructed* — explicitly, layer by layer, by a procedure that converges. Hilbert was right that the existence statement is true. Brouwer was right that mathematics ought to construct what it asserts. The transformer, oblivious to the *Grundlagenstreit*, does both.

---

## 9. Predictions of the Closed Future Cone

A framework deserves the name only if it forbids something. CFC forbids:

**Concepts with cyclic group symmetry should produce closed (compact, $\pi_1 \neq 0$) representational manifolds.** This is the Engels–Karkada prediction made absolute: not "circles are sometimes observed for cyclic concepts" but "any concept whose underlying language admits a finite cyclic symmetry must produce a compact representational manifold with non-trivial fundamental group, in any sufficiently expressive model." Karkada et al. (2026) prove this analytically in the symmetric-co-occurrence setting; CFC predicts the result extends to all cyclic concepts in all trained transformers above a critical scale.

**The depth of a transformer should correlate with the number of fixed-point iterations needed to settle the representational manifold.** Tasks that require many iterations of an operator (deep arithmetic, multi-step reasoning, long-range dependency resolution) should benefit super-linearly from depth, until convergence of the underlying fixed-point iteration is reached. Beyond convergence, further depth is wasted. The "depth saturation" phenomenon observed in deep equilibrium models is the operational signature of this saturation.

**The light-cone structure of attention should constrain what fixed points are reachable.** Information confined to tokens outside the attention cone of a given output position cannot enter the fixed-point computation for that position. This makes the failure modes of long-context reasoning (the "lost in the middle" phenomenon) a *causal-structure* failure: the cone closes too quickly relative to the path length required to traverse the representational manifold.

**Tasks with linear-time structure should produce ramping representations; tasks with cyclic-time structure should produce loop representations.** The Jarne et al. dichotomy is not a quirk of RNNs; it is a consequence of the topology of the underlying $\mathcal{M}$, and the same dichotomy should appear in transformer activations as the underlying task structure varies.

**Two concepts with the same underlying symmetry group should produce homeomorphic manifolds, regardless of semantic content.** The days-of-week circle and the months-of-year ring are both $S^1$; if CFC is right, the *cyclic minutes-on-a-clock concept*, the *seasons concept*, the *cardinal-directions concept* (N/E/S/W with $\mathbb{Z}_4$ symmetry), and any other $\mathbb{Z}_n$-symmetric concept should all produce $S^1$ — the only differences being the embedding angle and the position of the $n$ atoms on the loop.

---

## 10. What CFC Does Not Claim

It does not claim transformers compute *only* by fixed-point iteration. The forward pass has a finite number of layers; one can think of it as an *unrolled* fixed-point iteration, not a *converged* one. The fixed-point picture is the limit object, the topological invariant; the actual computation is finite.

It does not claim every concept has a non-trivial representational manifold. Concepts without symmetry will produce simply connected (disc-like) manifolds, in which case the LRH and MRH descriptions are exactly correct. CFC says LRH and MRH are the trivial-topology limits of a more general object.

It does not claim that the Brouwer fixed-point theorem in its classical (non-constructive) form is necessary for the analysis. The relevant tool is the constructive version: Banach's fixed-point theorem under contraction, or the Leray–Schauder principle for compact perturbations of the identity. The conceptual point — that compactness is the precondition under which the forward pass terminates at a well-defined answer — survives intact regardless of which formal version one prefers.

It does not claim the Minkowski analogy is metric, only causal. The attention pattern induces a causal cone structure (a partial order on token-layer events), not a metric of definite signature. There is no claim of a Lorentz metric on the residual stream space itself; the claim is that the *information-flow structure* of the architecture realizes the *causal* (not metric) content of Minkowski geometry.

It does not claim the Brouwer–Hilbert controversy is resolved in any technical sense. The point is rhetorical and conceptual: the transformer is a striking case where a classical existence theorem and its constructive counterpart are realized by the *same* concrete computation, in a way neither Brouwer nor Hilbert would have expected.

It does not claim novelty for any of the individual ingredients. Engels et al. (2024) found the circles. Karkada et al. (2026) proved the analytical reason. Jarne et al. (2025) found the corresponding RNN signatures. Universal-approximation-by-fixed-point-iteration is an established line. The synthesis — that all of this describes a *compact causally structured manifold with non-trivial topology, on which forward passes are constructive fixed-point iterations* — is the contribution.

---

## 11. A Final Thought Experiment

Imagine someone hands you a transformer. You do not know what it was trained on. You are not allowed to query it. You are allowed only to look at the geometry of its activations.

What can you read off the geometry?

The dimensionality of each concept's representational manifold tells you the rank of the underlying symmetry group. The topology of each manifold tells you which symmetries are cyclic (closed loops), which are translational (open lines), which are spherical (cooperative axes). The depth at which each manifold first stabilizes tells you how many fixed-point iterations of the operator the concept required. The shape of the light cone at each layer tells you the *causal range* of computation at that depth — how far back into the context the model can reach to assemble its answer.

You will not know what the model thinks. But you will know what kind of *world* its training corpus lived in, because the geometry of the corpus's symmetries is now permanently and visibly impressed onto the geometry of the model's activations. Karkada et al. (2026) proved this is what must happen. Engels et al. (2024) found the first instances. The Closed Future Cone is the framework that lets you generalize. The world is encoded in the topology.

A model that has lived among periodic things has rings in its activations. A model that has lived among hierarchical things has trees. A model that has lived among things with discrete reflection symmetries has bipartite graphs. A model that has lived among things with continuous symmetries has spheres and tori. None of this is metaphorical. All of it is a direct consequence of two ideas — Minkowski's causal structure of information flow, and Brouwer's topological structure of fixed points — applied to the kind of compact, causally constrained, iterated computation that a transformer performs.

The data is curved. The computation is causal. The answer is a fixed point. The representation is closed.

---

## 12. Sources

**The two organizing ideas**
- Minkowski, H. *Raum und Zeit.* Jahresbericht der Deutschen Mathematiker-Vereinigung **18**, 75–88, 1909.
- Einstein, A. *Zur Elektrodynamik bewegter Körper.* Annalen der Physik **17**, 891–921, 1905.
- Brouwer, L. E. J. *Über Abbildung von Mannigfaltigkeiten.* Mathematische Annalen **71**, 97–115, 1911.
- Hilbert, D. *Die Grundlagen der Mathematik.* Abhandlungen aus dem Mathematischen Seminar der Hamburgischen Universität **6**, 65–85, 1928.
- van Stigt, W. P. *Brouwer's Intuitionism.* North-Holland, 1990. [Brouwer–Hilbert controversy.]

**Causal circular representations in language models**
- Engels, J., Liao, I., Michaud, E. J., Gurnee, W., Tegmark, M. *Not All Language Model Features Are Linear.* NeurIPS 2024, arXiv:2405.14860.
- Karkada, D., Olah, C., Tegmark, M., et al. *Symmetry in language statistics shapes the geometry of model representations.* arXiv:2602.15029, February 2026.
- Modell, S. et al. 2025; Park, K., Choe, Y. J., Veitch, V. 2025; Prieto et al. 2026 — circular and toroidal representations across decoder-only language models.

**Time representation in recurrent dynamics**
- Jarne, C., Yoon, R., Eissa, T., Kilpatrick, Z. P., Josić, K. *Task-Parametrized Dynamics: Representation of Time and Decisions in Recurrent Neural Networks.* bioRxiv:2025.09.15.676356, September 2025.

**Transformers as fixed-point operators**
- Bai, S., Kolter, J. Z., Koltun, V. *Deep Equilibrium Models.* NeurIPS 2019.
- Hu, J. Y.-C., Liu, H., Chen, H.-Y., Wu, W., Liu, H. *Universal Approximation with Softmax Attention.* arXiv:2504.15956, updated December 2025.
- Li, G., Jiao, Y., Huang, Y., Wei, Y., Chen, Y. *Transformers Meet In-Context Learning: A Universal Approximation Theory.* arXiv:2506.05200, 2025.
- *Universal Approximation of Operators with Transformers and Neural Integral Operators* (Leray–Schauder construction). arXiv:2409.00841, 2024–2025.

**Earlier hypotheses CFC subsumes as limit cases**
- Park, K., Choe, Y. J., Veitch, V. *The Linear Representation Hypothesis and the Geometry of Large Language Models.* ICML 2024, arXiv:2311.03658. [LRH = local linearization of CFC.]
- Fel, T., Wang, T., Lepori, L., Kowal, M., Lee, S., Balestriero, R., Joseph, J., Lubana, E., Konkle, T., Ba, J., Wattenberg, M. *Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry.* arXiv:2510.08638, 2025–2026. [MRH = trivial-topology case of CFC.]

**Empirical grounding for the underlying compact structure**
- Robinson, J., Dey, M., Sweet, T. *The structure of the token space for large language models.* arXiv:2410.08993, 2024.
- Robinson, J., Dey, M., Chiang, K.-Y. *Token Embeddings Violate the Manifold Hypothesis.* arXiv:2504.01002, 2025.
- *TokenBlowUp: Resolving Representational Singularities.* arXiv:2507.19747, 2025.

**Goodfire interpretability programme**
- Goodfire Research. *Steering Along Manifolds to Control Neural Networks.* Goodfire blog, 2026.
- Goodfire Research. *The World Inside Neural Networks.* Goodfire blog, 2026.

---

## 13. Summary

Hermann Minkowski took relativity and made it geometric: time is a coordinate, signature distinguishes timelike from spacelike, causation is the light cone. L. E. J. Brouwer proved that compactness plus continuity gives you a fixed point — and then, decades later, spent the rest of his career arguing such proofs should never have been believed without a construction.

Deep networks, a hundred years after, sit at the intersection. The information flow through a transformer's layers traces a *light cone* whose causal structure determines what reaches what. The representations along which the network computes live on *compact* manifolds — provably, by Karkada et al. (2026), for any concept whose underlying language has a finite cyclic symmetry — whose topology is non-trivial whenever the symmetry is non-trivial. The forward pass is the *constructive* realization of Brouwer's *non-constructive* fixed-point theorem: each layer is one iteration; the answer the model produces is the iterate at the final layer; the topology of the representational manifold is what makes that answer well-defined under modular shifts.

The Engels et al. circle of days, the Karkada et al. analytical proof that symmetry forces such circles, the Jarne et al. dichotomy between cyclic and ramping time representations in RNNs, the universal-approximation-via-Brouwer-style-fixed-points results in transformers, the LRH and MRH descriptions of representational geometry — these are all describing the same object from different angles. The Linear Representation Hypothesis is its local linearization. The Minkowski Representation Hypothesis is its trivial-topology limit. The Closed Future Cone is the general case.

The data was curved. The data was cyclic. The data lived inside a light cone of causation that the architecture inherited and the training preserved. The model walked the fixed point into being, and the loops it walked around are the topological invariants the world impressed on it. The geometry was always there. The framework — finally — is its name.
