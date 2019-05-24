extern crate dynet_sys as dn;

use super::{Parameter, Expression};

/// Computation graph structure.
/// 
/// `ComputationGraph` is central to the inner workings of DyNet where nodes
/// represent forward and backward intermediate values, and edges represent
/// functions of multiple values. From the
/// [Dynet technical report](https://arxiv.org/abs/1701.03980): 
/// 
/// > [The] computation graph represents symbolic computation, and the results
/// > of the computation are evaluated lazily: the computation is only performed
/// > once the user explicitly asks for it (at which point a “forward”
/// > computation is triggered). Expressions that evaluate to scalars (i.e. loss
/// > values) can also be used to trigger a “backward” computation, computing
/// > the gradients of the computation with respect to the parameters.
/// 
/// 
pub struct ComputationGraph {
    inner: *mut dn::DN_ComputationGraph
}


impl_new!(ComputationGraph, DN_NewComputationGraph, "Create a new computation graph. Call this before building any new computation graph");
impl_drop!(ComputationGraph, DN_DeleteComputationGraph);

impl ComputationGraph {
    /// Load parameters into the computation graph and returns an parameter
    /// `Expression` which can be used to build computation graph later.
    pub fn load_param(&self, p: &Parameter) -> Expression {
        unsafe {
            Expression{inner: dn::DN_LoadParamToCG(self.inner, p.inner)}
        }
    }


    /// Add input to the computation graph and return a an expression that
    /// represents a vector, matrix, or tensor input.The returned `Expression`
    /// object can be used to build computation graph later.
    /// 
    /// The dimensions of the input are defined by `dim`.
    /// 
    /// `vals` of type `&[f32]` should contain the values used to fill the input
    /// , in column-major format. The length must equal to the product of all
    ///  dimensions in `dim`.
    /// 
    /// # Example
    /// ```
    /// let cg = dynet::ComputationGraph::new();
    /// let vec_vals = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    /// let matrix_vals = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    /// 
    /// // Create a 5-length vector input.
    /// let x = cg.add_input(&vec_vals, &[5]);
    /// 
    /// // Create a 2x3 matrix input.
    /// let y = cg.add_input(&matrix_vals, &[2, 3]);
    /// ```
    pub fn add_input(&self, vals: &[f32], dim: &[i64]) -> Expression {
        unsafe {
            let dim_ptr = dn::DN_NewDimFromArray(dim.as_ptr(), dim.len(), 1);
            Expression{inner: dn::DN_AddInputToCG(
                self.inner, dim_ptr, vals.as_ptr(), vals.len()
            )}
        }
    }
    
    ///  
    /// 
    /// The last dimension of the `dim` is used as the batch dimension.
    pub fn add_batched_input(&self, vals: &[f32], dim: &[i64]) -> Expression {
        unsafe {
            let dim_ptr = dn::DN_NewDimFromArray(
                dim.as_ptr(),
                dim.len() - 1,
                dim[dim.len() - 1] as u32);
            Expression{inner: dn::DN_AddInputToCG(
                self.inner, dim_ptr, vals.as_ptr(), vals.len()
            )}
        }
    }

    /// Run complete forward pass from first node to given one, ignoring all 
    /// precomputed values.
    pub fn forward(&self, last: &Expression) -> f32 {
        unsafe {
            dn::DN_Forward(self.inner, last.inner)
        }
    }

    /// 
    pub fn backward(&self, last: &Expression) {
        unsafe {
            dn::DN_Backward(self.inner, last.inner, false);
        }
    }

    /// Print the computation graph directly to the stdout by graphviz dot syntax.
    /// 
    /// Used for debugging.
    pub fn print_graphviz(&self) {
        unsafe {
            dn::DN_PrintGraphviz(self.inner);
        }
    }

    /// Get the unique graph ID.
    /// 
    /// This ID is incremented by 1 each time a computation graph is created.
    pub fn id(&self) -> u32 {
        unsafe {
            dn::DN_GetCGId(self.inner)
        }
    }

    /// Set a checkpoint.
    pub fn set_checkpoint(&self) {
        unsafe {
            dn::DN_SetCGCheckPoint(self.inner);
        }
    }

    /// Revert to last checkpoint.
    pub fn revert(&self) {
        unsafe {
            dn::DN_RevertCG(self.inner);
        }
    }
}


/// Get id of the current active graph.
/// 
/// This can help check whether a graph is stale.
pub fn get_current_graph_id() -> u32 {
    unsafe {
        dn::DN_GetCurrentGraphId()
    }
}




