#![allow(non_snake_case)]
extern crate dynet as dy;

fn main() {
    dy::initialize();
    let m = dy::ParameterCollection::new();

    let vocab_size = 100;
    let dim = 10;
    let lp = m.add_default_lookup_param(&[vocab_size, dim]);

}