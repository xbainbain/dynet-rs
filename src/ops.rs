extern crate dynet_sys as dn;

use super::Expression;

pub fn tanh(x: &Expression) -> Expression {
    unsafe {
        Expression{inner: dn::DN_Tanh(x.inner)}
    }
}


pub fn squared_distance(x: &Expression, y: &Expression) -> Expression {
    unsafe {
        Expression{inner: dn::DN_SquaredDistance(x.inner, y.inner)}
    }
    
}

pub fn sum(xs: &[Expression]) -> Expression {
    let mut xs_ptr:Vec<*mut dn::DN_Expression> = xs.iter().map(|x| x.inner).collect();
    unsafe {
        Expression{
            inner: dn::DN_Sum(xs_ptr.as_mut_slice().as_mut_ptr(), xs_ptr.len() as i32)
        }
    }
}

pub fn sum_batches(x: &Expression) -> Expression {
    unsafe {
        Expression{inner: dn::DN_SumBatches(x.inner)}
    }
}