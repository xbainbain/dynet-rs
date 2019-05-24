# dynet-rs
Rust bindings for DyNet. By using dynet-rs, you can directly write your nerual network in Rust.
> DyNet is a neural network library developed by Carnegie Mellon University and many others. It is written in C++ (with bindings in Python)

This projet is still in progress and not stable yet. (And for this reason I haven't put it on [crate.io](https://crates.io) by now which is Rust official package registry). But you can still use it have fun for now.

# Examples
## Writing a simple network to solve x-or problem
Define parameters of network and the optimizer(trainer) for it.
```rust
let pc = dy::ParameterCollection::new(); 
let p_W = pc.add_default_param(&[HIDDEN_SIZE, 2]);
let p_b = pc.add_default_param(&[HIDDEN_SIZE]);
let p_V = pc.add_default_param(&[1, HIDDEN_SIZE]);
let p_a = pc.add_default_param(&[1]);

let trainer = SimpleSGD::default(&pc);
```
Create computation graph.
```rust
let cg = dy::ComputationGraph::new();

// Load parameter to the cg
let W = cg.load_param(&p_W);
let b = cg.load_param(&p_b);
let V = cg.load_param(&p_V);
let a = cg.load_param(&p_a);

// Add input to the cg
let x_values = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0];
let y_values = [-1.0, 1.0, 1.0, -1.0];
let x = cg.add_batched_input(&x_values, &[2, 4]);
let y = cg.add_batched_input(&y_values, &[1, 4]);

// Build cg
let h = tanh(&(&(&W * &x) + &b));
let y_pred = &(&V * &h) + &a;
let loss = squared_distance(&y_pred, &y);
let sum_loss = sum_batches(&loss);
```
Train the neural network.
```rust
for iter in 0..ITERATION {
    let my_loss = cg.forward(&sum_loss) / 4.0;
    cg.backward(&sum_loss);
    trainer.update();
    println!("iter{}: loss = {}", iter+1, my_loss);
}
```

Or you directly run this example by:
```bash
cargo run --example x_or_batch
```

# License
*dynet-rs* is distributed under the terms of the MIT license.
