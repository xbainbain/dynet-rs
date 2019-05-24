extern crate dynet_sys as dn;

use std::ops::{Index};
/// The Dim struct stores information on the shape of a tensor.
/// 
/// In DyNet the dimensions are represented as the **standard dimension + the
/// batch dimension**, which makes batched computation transparent.
/// 
/// 
pub struct Dim {
    inner: *mut dn::DN_Dim,
    d: Vec<u32>
}

////////////////////////////////////////////////////////////////////////////////
// Inherent methods
////////////////////////////////////////////////////////////////////////////////
impl_drop!(Dim, DN_DeleteDim);

impl Dim {
    /// Initialize from an slice of dimensions and a batch size
    pub fn new(dims: &[u32], batch_size: u32) -> Self {
        let mut d = Vec::new();
        for d_val in dims {
            d.push(*d_val);
        }


        let cast_dims:Vec<i64> = dims.iter().map(|d| *d as i64).collect();
        let dims_ptr = cast_dims.as_ptr();
        let num_dims = cast_dims.len();
        
        unsafe {
            Dim{
                inner: dn::DN_NewDimFromArray(dims_ptr, num_dims, batch_size),
                d: d
            }
        }
    }

    /// Get the size(value) of the batch dimension.
    pub fn batch_size(&self) -> u32 {
        unsafe {
            dn::DN_DimBatchElems(self.inner)
        }
    }

    /// Get the ordre of the dimension.
    pub fn ordre(&self) -> u32 {
        unsafe {
            dn::DN_DimNumDim(self.inner)
        }
    }
}

impl Index<usize> for Dim {
    type Output = u32;
    fn index(&self, index: usize) -> &u32 {
        &self.d[index]
    }
}