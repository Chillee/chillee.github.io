# How Many Paths of Length K are there? : And I Show You How Deep the Rabbit Hole Goes
Here's a programming problem: Given a directed unweighted graph with V vertices and E edges, how many paths of length K are there from node A to node B? Paths may visit the same node or edge multiple times. [^walkpath]

```dot {engine="dot"}
digraph Figure {
    label = "There are 2 paths of length 2 from node A to node B"
    labelloc=top
    node[label=""];
    subgraph cluster_first {
        label="Graph";
        A1 [label="A"]
        B1 [label="B"]
        A1 -> C1 -> B1
        A1 -> D1 -> B1
    }
    subgraph cluster_second {
        label="Path 1";
        A2 [label="A"]
        B2 [label="B"]
        A2 -> C2 -> B2[color=red, penwidth=3.0]
        A2 -> D2 -> B2
    }
    subgraph cluster_third {
        label="Path 2";
        A3 [label="A"]
        B3 [label="B"]
        A3 -> C3 -> B3
        A3 -> D3 -> B3[color=red, penwidth=3.0]
    }
}
```

This problem is fairly standard -  many of you may have seen it or heard it in an interview. Personally, I've seen this problem on Hackernews in some form at least three times, [here](https://news.ycombinator.com/item?id=22953404), [here](https://news.ycombinator.com/item?id=19193405), and [here](https://news.ycombinator.com/item?id=17758800).

I found this comment from the last link particularly interesting.
> I also work at Google... I used to love asking this question until it was banned.

The commenter proceeds to outline several different levels of solutions. His levels are:
1. A brute force exponential solution
2. A dynamic programming solution in $O(EK)$
3. A logarithmic time solution involving adjacency matrices, matrix multiplication, and binary decomposition of numbers. This solution is $O(V^3 \log K)$.

He states of the most advanced level:
> I've never seen anyone get this and I only learned about it after months of asking this question.

And that's where he left the problem. Seems like a pretty cool problem with plenty of depth, doesn't it?

What if I told you that this problem has even more depth than you thought? What if I told you that, drawing on concepts from coding theory, abstract algebra, and signal processing, we could solve this problem in $O(EV + V \log V \log K)$ time?

To my knowledge, despite being a common problem, I have not seen this faster solution presented anywhere.

Let's dive down the rabbit hole of solutions to this problem. We'll start with the solutions that have already been discussed, but then...


## The Obvious Brute Force Solution ($V \leq 5, K\leq 5$)
The most straightforward solution to this problem is to enumerate all the paths and stopping once our path reaches K nodes. We can implement it with a breadth first search like so:

```python
ans = 0
queue = [(A, 0)] # We track (length of walk, current node)
while not queue.empty():
    curNode, curLength = queue.front()
    queue.pop()
    if curLength == K:
        if curNode == B:
            ans += 1
        break
    for neighbor in neighbors(curNode):
        queue.push((neighbor, curLength + 1))
```
However, this solution is exponential. Take this simple graph

```dot
digraph G {
    0 -> 1
    1 -> 0
    0 -> 0
    1 -> 1
}
```

Let's count the number of paths of length K from node 0 to node 1. We see that any sequence of 0s and 1s that ends at node 1 is a valid path, implying that there are $2^K$ valid paths.

Thus, this solution is exponential. An interviewer (like that Googler) wouldn't be too impressed.

## Dynamic Programming ($E \leq 5000, K \leq 5000$)
Looking at the above solution, we notice that there a lot of wasted work. Our queue will often visit the exact same state many times. For example, we'll visit node B with a path of length K as many times as our answer. Noticing that the same state is visitied multiple times naturally leads to a dynamic programming solution.

In this case, we choose the same state as we did in our above problem: (node, length). This time, however, we consolidate all of our redundant states.

Thus, `dp[node][length] = sum(dp[neighbors(node)][length-1])`.


```python
dp[A][0] = 1
for length in 0..K:
    for node in 0..N:
        for neighbor in neighbors(node):
            dp[node][length] += dp[neighbor][length-1]
```

We can either do this iteratively (which allows us to reduce memory complexity by only storing one layer of DP at a time), or recursively with memoization. I've presented the iterative solution above.

The complexity of this solution is $O(EK)$. So far, this seems like a standard DP problem. Most interviewers would probably be satisfied with this solution. But we aren't.

## A Neat Trick With Adjacency Matrices ($V \leq 500, K \leq 1 \text{ billion}$)
If there aren't many nodes, but $K$ is extremely large, then we need to give up on the above DP approach. The naive above finds the answer for "how many paths of length $K-1$" before it finds the answer to "how many paths of length $K$". As a result, even if we *could* find the answer for $K$ from the answer for $K-1$ in constant time, we wouldn't be able to solve this problem with the above constraints.

There's 2 ways to proceed. The first is to note that we don't actually *need* the answer for $K-1$ before we can find the answer for $K$. For example, if we know that that there are 3 paths of length 50 from A to C and 4 paths of length 50 from C to B, then there are $3 \cdot 4$ paths of length 100 from A to C to B. More concretely, consider any node C. The number of paths from A to B of length $K$ that include C at the midpoint is the number of paths from A to C of half length multiplied by the number of paths from C to B of half length. If we sum over all possible nodes for C, then we have our answer for $K$.

```dot
digraph G {
    label = "12 paths from A to B of length 100"
    labelloc=top
    A -> C
    A -> C
    A -> C
    C -> B
    C -> B
    C -> B
    C -> B
}
```

This allows us to remove our linear dependence on $K$ and transforms it into a logarithmic one, for a $ V^3 \log K$ algorithm.

Using graph theory, however, there's an even easier way to come up with this algorithm. One representation for graphs is as an adjacency matrix $A$. If one views the values in $A_{ij}$ as the number of edges between $i$ and $j$, then $A_{ij}^k$ represents the number of paths between $i$ and $j$ of length $k$.

Thus, this problem has been reduced to computing $A^k$ (i.e: matrix power). This is a standard problem that can be done in $V^3 \log K$ time through [binary exponentiation](https://news.ycombinator.com/item?id=22946710).

The second approach can be thought of as an abstraction of the first approach to standard concepts like matrix exponentiation.


## Going Even Deeper... ($V \leq 10000, E \leq 10000, K \leq 1 \text{ billion}$)
For the vast majority of people, their ability to solve this problem stops here. And thus far, I've covered nothing that existing articles haven't already done (see [this](http://www.math.ucsd.edu/~gptesler/184a/slides/184a_ch10.3slides_17-handout.pdf) or [this](https://www.geeksforgeeks.org/count-possible-paths-source-destination-exactly-k-edges/)).

This is also where the previously mentioned HN commenter thought the problem ended.

To go even deeper, we must first take a detour into linear recurrences and generating functions.

### Linear Recurrences
A linear recurrence is a recurrence like: $a_k = 3a_{k-1} + 2a_{k-2} - a_{k-3}$. The Fibonacci series is a famous linear recurrence, which can be written as $A_k = A_{k-1} + A_{k-2}$. The order of a linear recurrence is the number of terms it depends on. So, the first example would have order 3, and Fibonacci would have order 2.

You might know that finding the $K$-th Fibonacci number can be done in $\log(K)$ time using matrix exponentiation. In fact, you can find the $K$-th term of any order $N$ linear recurrence using matrix exponentiation in $O(N^3 \log K)$ time. This is a [good resource if you're unfamiliar](https://community.topcoder.com/tc?module=Static&d1=features&d2=010408). This is simply an extension of Fast Fibonacci methods like [those found here](https://www.nayuki.io/page/fast-fibonacci-algorithms).

However, if we're working with linear recurrences, there's another (somewhat unintuitive) form that allows for even faster computation.

#### Polynomials and Generating Functions
Let's define a (weird) function $G$, which takes in any polynomial and replaces $x^k$ with the $k$-th term in our linear recurrence. So, for Fibonacci,
$G(x^0)=1$, $G(x^1) = 1$, $G(x^2) = 2$, $G(x^3)=3$, $G(x^4)=5$, $G(x^5)=8$, and $G(x^k) = k$-th Fibonacci element. We can also pass in more than one term, so $G(x^2 + x^3) = G(x^2) + G(x^3) = 5$. Finally, $G$ is also a linear function, which means that $G(f+g) = G(f) + G(g)$.

Some more examples:
\[G(x(x^2 + 2x^3)) = G(x^3 + 2x^4) = A_3 + 2A_4 = 3 + 2\cdot 5 = 13 \]
\[G(x^{20} + 3) = A_{20} + 3A_0 = 6765 + 3 = 6768\]

So, if someone gave us a magical black box to evaluate $G$, we could simply evaluate $G(x^k)$ and get our answer! Unfortunately, we don't. If $K$ was small enough, we could compute the terms up to $k$ ourselves. But since $K$ is extremely large, that approach isn't feasible.

Another way we could evaluate $G(x^k)$ is to find a polynomial that was equivalent to $G(x^k)$ and evaluating that instead. And if this polynomial had low degree, then evaluating this function would be easy.

For example, this is one way to find an equivalent polynomial of lower degree:
\[G(x^5) = G(x^3 + x^4) = G(x^3 + (x^2 + x^3)) = G(2x^2 + 3x^3) = 2F_2 + 3F_3 = 8\]

So, if we could easily compute a polynomial $h$ with low degree such that $G(x^k) = G(h)$, we would be done!

#### Annihilation

Before linear recurrences become useful, we need to introduce one more concept: the ominously named "annihilator" polynomial. An annihilator is a non-zero polynomial $f$ such that $G(f) = 0$. On Fibonacci, for example, examples of annihilators would be $G(x^3 - x^2 - x^1)$ or $G(x^6 - x^5 - x^4)$. Remember that $G$ turns polynomial terms into Fibonacci terms. So, another way to view this last annihilator is that it says: "the 6th Fibonacci term is equal to the 5th and 4th Fibonacci terms added together".

This last statement is clearly true. After all, that's the definition of Fibonacci.

This observation leads to an easy way of generating annihilators: We just use the definition of our linear recurrence! If the n-th term of our linear recurrence is some combination of the previous terms, then the n-th term minus those previous terms is equal to $0$.

For illustration, let's take the linear recurrence $a_n = a_{n-1} - 2a_{n-2} + 3a_{n-3}$. Thus, $a_3 = a_2 - 2a_1 + 3a_0$. This implies that $a_3 - a_2 + 2a_1 - 3a_0 = 0$. Thus, one annihilator is $x^3 - x^2 + 2x - 3$. We can repeat this process with $a_4$ to get the annihilator $x^4 - x^3 + 2x^2 - 3x$ or $a_{100}$ to get the annihilator $x^{100} - x^{99} + 2x^{98} - 3x^{97}$. Note that we can also generate these other annihilators by multiplying the annihilator for $a_3$ by $x^n$.

Since $G(fx) = G(fx^2) = G(fx^3)  = ...= 0$, this means that $G(fg) = 0$, where $g$ can be any polynomial. For example, $G(f(x^3 + x^7)) = G(fx^3 + fx^7) = G(fx^3) + G(fx^7) = 0$.

#### Computing the K-th term of a linear recurrence with polynomials
Now, we know how to represent linear recurrences with generating functions, and we even know what an annihilator is. But how does that allow us to do anything useful?

Let's take a short digression into normal integer arithmetic. For any integer $a$ and $b$ where $a < b$, we can write $b$ as $d\cdot a + (b \mod a)$ for some integer $d$. For example, for $a=7$ and $b=30$, we can write $30 = 4\cdot 7 + (30 \mod 7)$.

We can apply a similar concept to polynomials [^2]. For any polynomial $a$ and $b$ where $a$ has lower degree than $b$, we can write $b$ as $d\cdot a + (b\mod a)$, where $d$ is some polynomial.

Now, this is where it all comes together. Let's plug in $a = f$ (our annihilator) and $b = x^k$ (the term we're looking for).

\[x^k = d\cdot f + (x^k \mod f)\]

We don't know what $d$ is, but remember that $f$ is an annihilator. Thus, no matter what $d$ is, $G(d\cdot f) = 0$ and we can ignore it! Thus, we know that $G(x^k) = G(x^k \mod f)$. As $x^k \mod f$ is a polynomial with low degree, we can evaluate $G(x^k \mod f)$ easily.

Calculating $x^k % f$ can be done with $\log k$ polynomial multiplication and polynomial modulo. Using FFT-based operations, we can do each of these in $\log n$ time. Thus, we can find the $K$-th term of an order $N$ linear recurrence in $N \log N \log K$ time.

### Connecting back to the original problem with Cayley Hamilton
As mentioned previously, we know that finding the $k$-th term of any linear recurrence can be reduced to finding the $k$-th power of a matrix. But how do we know that the inverse is true: that finding the $k$-th power of a matrix can be reduced to finding the $k$-th term of a linear recurrence? This is the missing link for applying linear recurrences to our problem.

[Wikipedia](https://en.wikipedia.org/wiki/Cayley%E2%80%93Hamilton_theorem) writes that

> If A is a given n×n matrix and $I_n$  is the n×n identity matrix, then the characteristic polynomial of A is defined as
> $${\displaystyle p(\lambda )=\det(\lambda I_{n}-A)~}$$

Not immediately helpful. However, several lines down we see that
> The [Cayley Hamilton] theorem allows A^n to be expressed as a linear combination of the lower matrix powers of A.

In other words, we know that this equation holds true for some values of $x_i$.

$$A^n = x_0I + x_1A + x_2A^2 ...$$

In other words, we are **guaranteed** that the powers of $A$ form a linear recurrence! This is not obvious at all, but it does highlight some of the powers of math. Who would have thought that what seemed like a simple DP problem would connect with a fundamental linear algebra theorem?

Although it's neat that a linear recurrence exist, how do we actually recover this linear recurrence from the matrix powers? Enter a fairly obscure algorithm from coding theory.

### Berlekamp-Massey

Wikipedia states that:

> The Berlekamp–Massey algorithm is an algorithm that will find the shortest linear feedback shift register (LFSR) for a given binary output sequence.

I couldn't tell you what a Linear Feedback Shift Register is, but I **can** tell you what Berlekamp Massey does. Given a sequence of terms, it finds the shortest linear recurrence that fits all of the terms. If you pass in $2L$ terms, it's guaranteed that the shortest linear recurrence is of order $\leq L$. For example, if you pass $1, 1, 2, 3$ it will provide you the Fibonacci sequence. And it'll do it in $N^2$ time!

There's only one last wrinkle - Berlekamp Massey doesn't operate on matrices. However, note that Cayley-Hamilton doesn't merely imply that the matrices satisfy some linear recurrence, it also implies that each individual element satisfies a linear recurrence. For example, $A^n_{0,0} = x_0 + x_1A_{0,0} + x_2A^2_{0,0}...$

Thus, if we're trying to count the number of paths from node A to node B, we can simply pass $I_{a,b}, A_{a,b},A^2_{a,b}...$ to Berlekamp-Massey. As our linear recurrence is order $N$, we need to pass in the first $2N$ terms of $A^i$.

As our DP approach lets us find all the values of $A^i$ up to $A^k$ in $EK$ time, this allows us to compute all of the necessary terms in $EV$ time and the corresponding linear recurrence in $V^2$ time.

And that's it! Finally, we can compute the $k$-th term of the linear recurrence in $V \log V \log K$ time.

# Concluding Thoughts
Hope you found this a neat rabbit hole to dive into! I think this problem is one that many people may have encountered, whether in class or interviews. Although many of you may have written it off as a basic DP problem (or perhaps even after encountering the adjacency matrix interpretation), there's actually a lot deeper you can go with this problem.

Besides the problem-specific lessons, this is perhaps a lesson on how math can become quite helpful, even on problems where math originally didn't seem relevant. At the beginning, this problem seemed like your typical interview fare, but as we kept pushing the limits of this problem we ended up needing more and more math. In the end, we needed to start citing cutting-edge research papers.

### Acknowledgements
I'd like to thank the competitive programming discord AC, and in particular `pajenegod`, who both came up with this technique and explained it to me!

# Further Extensions
## Q: What if you wanted all the terms of $A^k$, instead of just one?
Our approach currently only provides us one of the $N^2$ terms. Naively, we could try using the linear recurrence we found for a single element and use it for all of the other terms. Unfortunately, this doesn't work. The most obvious counterexample is when the graph is disconnected. Obviously, the paths in one component provides no information about paths in a different connected component. However, even if we restrict ourselves to the case where the graph is connected, it's not guaranteed that a linear recurrence found for one element in $A$ will generalize to the whole matrix. For example, take this undirected graph <----insert graph---> The number of paths from 0->1 follow the sequence `0,1,0,2,0,4...` (a linear recurrence of order 2) while the number of paths from 2 -> 1 follow the sequence `0,0,1,0,2,0,4...`.

One intuition for why this doesn't work is that information about a single node isn't enough for the whole graph. Information might not reach that node, it might be cancelled out, etc. One thing we can try is to find the linear recurrence of $a\cdot A^n_{0,0} + b\cdot A^n_{1,1} = (aI_{0,0} + bI_{1,1}) + (aA_{0,0} + bA_{1,1})...$. In other words, we take a random linear combination of the elements of our matrix, and hope that this allows us to recover the linear recurrence for the whole matrix. But how well does this work?

As it turns out, the procedure we use here is actually very similar to something called the Coppersmith algorithm. Corollary 22 of [this paper](https://sci-hub.tw/https://www.sciencedirect.com/science/article/pii/S0747717115000528) states that the procedure outlined above succeeds with probability $(1-1/q)^{2n}$, where $q$ is the cardinality of our finite field (how big our mod is). Thus, with high probability our method will recover the linear recurrence for the whole graph!

We can actually use this method as a general finite matrix field exponentiation method (assuming that $q \gg n$). We can find $A^K$ for an arbitrary matrix $A$ in $V^3 + V \log V \log K$.

## Q: Why don't we just diagonalize the matrix and compute $A^k$ that way?
If the matrix was diagonalizable (e.g: undirected graph), we could - if we didn't care about floating point precision.  We can't solve this problem by diagonalizing your matrix over a finite field either. Symmetric matrices are not guaranteed to be diagonalizable over finite fields.

For directed graphs, we could also compute the Jordan Normal Form, but we run into the same floating point issues.


[^walkpath]: Some computer scientists will tell you that the correct terminology should be walk, but I suspect most programmers are more familiar with the term "path". Solving this problem for actual (simple) paths is NP-Complete, as setting $K=N$ reduces this problem to Hamiltonian Path.
[^2]: This is a general property of Euclidean rings.