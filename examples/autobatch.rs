#![allow(non_snake_case)]
extern crate dynet as dy;

use dy::trainer::Trainer;
use dy::ops::*;

static ITERATION:u32 = 30;
static HIDDEN_SIZE:u32 = 8;

fn main() {
    dy::initialize();

    let pc = dy::ParameterCollection::new();
    let p_W = pc.add_default_param(&[HIDDEN_SIZE, 2]);
    let p_b = pc.add_default_param(&[HIDDEN_SIZE]);
    let p_V = pc.add_default_param(&[1, HIDDEN_SIZE]);
    let p_a = pc.add_default_param(&[1]);

    println!("{}", pc.size());
    let trainer = dy::trainer::SimpleSGD::default(&pc);

    for iter in 0..ITERATION {
        let cg = dy::ComputationGraph::new();

        // Load parameter to the cg
        let W = cg.load_param(&p_W);
        let b = cg.load_param(&p_b);
        let V = cg.load_param(&p_V);
        let a = cg.load_param(&p_a);

        let mut losses = Vec::new();

        for mi in 0..4 {    
            let x1 = (mi % 2) > 0;
            let x2 = (mi / 2) % 2 > 0;
            let x_vals = [if x1 {1.0f32} else {-1.0f32}, 
                if x2 {1.0f32} else {-1.0f32}];
            let y_val = [if x1 != x1 {1.0f32} else {-1.0f32}];

            // Add inputs to cg
            let x = cg.add_input(&x_vals, &[2]);
            let y = cg.add_input(&y_val, &[1]);

            let h = tanh(&(&(&W * &x) + &b));
            let y_pred = tanh(&(&(&V * &h) + &a));
            losses.push(squared_distance(&y, &y_pred));
        }

        let loss_expr = sum(&losses);

        // Print the graph, just for fun.
        if iter == 0 {
            cg.print_graphviz();
        }

        // Calculate the loss. Batching will automatically be done here.
        let loss = cg.forward(&loss_expr) / 4.0;
        cg.backward(&loss_expr);
        println!("{}", pc.gradient_l2_norm());
        trainer.update();

        println!("loss = {}", loss);
    }
}