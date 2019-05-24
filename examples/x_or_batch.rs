#![allow(non_snake_case)]
extern crate dynet as dy;

use dy::trainer::*;
use dy::ops::*;

static HIDDEN_SIZE:i64 = 8;
static ITERATION:u32 = 200;

fn main() {
    dy::initialize();

    let pc = dy::ParameterCollection::new();
    
    let p_W = pc.add_default_param(&[HIDDEN_SIZE, 2]);
    let p_b = pc.add_default_param(&[HIDDEN_SIZE]);
    let p_V = pc.add_default_param(&[1, HIDDEN_SIZE]);
    let p_a = pc.add_default_param(&[1]);

    let trainer = SimpleSGD::default(&pc);

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

    // Train the parameters
    for iter in 0..ITERATION {
        let my_loss = cg.forward(&sum_loss) / 4.0;
        cg.backward(&sum_loss);
        trainer.update();
        println!("iter{}: loss = {}", iter+1, my_loss);
    }
}