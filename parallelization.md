How to Train Very Large Deep Learning Models from First Principles

1. Considerations
Memory Usage
Communication
Computation
Batch Sizes

1. Memory usage
Parameter Memory
Optimizer Memory
Gradient Memory
Activation Memory (i.e. save for backwards pass)
    - Complicated, but rule of thumb for transformers is sqrt(params) * seq_length * batch_size

Methods of Reducing Memory Usage:
Data Parallelism (activation memory)
Gradient Checkpointing/Rematerialization (activation memory)
    RevNets
Zero 1 (optimizer memory)
Zero 2 (gradient memory)
Zero 3 (parameter memory)


2. Model-Parallelism
3. Overlapping
