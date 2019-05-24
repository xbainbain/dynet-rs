extern crate dynet as dy;

#[test]
fn weight_decay_lambda() {
    let pc = dy::ParameterCollection::with_weight_decay(0.5);
    assert_eq!(0.5, pc.weight_decay_lambda());
}

