extern crate dynet_sys as dn;

use super::{ParameterCollection};

/// Optimizers that can be used to turn parameters.
pub trait Trainer {
    /// Create a trainer with the default values for the superparameters.
    fn default(pc: &ParameterCollection) -> Self;

    /// Update the parameters according to the appropriate update rule.
    fn update(&self);

    /// Clip gradient.
    /// 
    /// If clipping is enabled and the gradient is too big, return the amount to
    /// scale the gradient by (otherwise 1).
    fn clip_gradients(&self) -> f32;
}

////////////////////////////////////////////////////////////////////////////////

/// Stochastic gradient descent trainer.
/// 
/// This trainer performs stochastic gradient descent, the goto optimization
/// procedure for neural networks.
pub struct SimpleSGD {
    inner: *mut dn::DN_SimpleSGDTrainer
}

impl SimpleSGD {
    /// Create a stochastic gradient descent trainer with the initial learning 
    /// rate `lr`.
    pub fn new(pc: &ParameterCollection, lr: f32) -> Self {
        unsafe {
            SimpleSGD{inner: dn::DN_NewSimpleSGDTrainer(pc.inner, lr)}
        }
    }
}

impl_drop!(SimpleSGD, DN_DeleteSimpleSGDTrainer);

impl Trainer for SimpleSGD {
    fn default(pc: &ParameterCollection) -> Self {
        unsafe {
            SimpleSGD{inner: dn::DN_NewSimpleSGDTrainer(pc.inner, 0.1)}
        }
    }

    fn update(&self) {
        unsafe {
            dn::DN_SimpleSGDTrainerUpdate(self.inner);
        }
    }

    fn clip_gradients(&self) -> f32 {
        unsafe {
            dn::DN_SimpleSGDTrainerClipGradients(self.inner)
        }
    } 
}

////////////////////////////////////////////////////////////////////////////////


