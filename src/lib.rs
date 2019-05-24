extern crate dynet_sys as dn;

use std::ffi::CString;
use std::ffi::CStr;

use std::ops::{Add, Mul};

////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////
macro_rules! impl_new {
  ($name: ident, $call:ident, $doc:expr) => {
    impl $name {
      #[doc = $doc]
      pub fn new() -> Self {
        unsafe {
          let inner = dn::$call();
          assert!(!inner.is_null());
          $name {
            inner: inner,
          }
        }
      }
    }
  }
}

macro_rules! impl_drop {
  ($name: ident, $call:ident) => {
    impl Drop for $name {
      fn drop(&mut self) {
        unsafe {
          dn::$call(self.inner);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Parameter
////////////////////////////////////////////////////////////////////////////////

/// Parameters are things that are optimized during training.
/// 
/// Use `ParameterCollection`'s method `` to create a new `Parameter` and add it to
/// the parameter collection in the same time.
pub struct Parameter {
    inner: *mut dn::DN_Parameter
}

impl_drop!(Parameter, DN_DeleteParameter);

impl Parameter {
    pub fn print_values(&self) {
        unsafe {
            let val_tensor = dn::DN_ParameterValues(self.inner);
            dn::DN_PrintTensor(val_tensor)
        }
    }


}


////////////////////////////////////////////////////////////////////////////////
// LookupParameter
////////////////////////////////////////////////////////////////////////////////

/// 
pub struct LookupParameter {
    inner: *mut dn::DN_LookupParameter
}

impl_drop!(LookupParameter, DN_DeleteLookupParameter);

impl LookupParameter {

}






////////////////////////////////////////////////////////////////////////////////
// ParameterCollection
////////////////////////////////////////////////////////////////////////////////

///A ParameterCollection is a container for Parameters and LookupParameters.
/// 
/// Use it to create, load and save parameters.
/// 
/// The values of the parameters in a collection can be persisted to and loaded 
/// from files.
pub struct ParameterCollection {
    inner: *mut dn::DN_ParameterCollection
}

impl_new!(ParameterCollection, DN_NewParameterCollection, "Create a new ParameterCollection. Weight-decay value is taken from commandline option.");
impl_drop!(ParameterCollection, DN_DeleteParameterCollection);

impl ParameterCollection {
    /// Constructs a new `ParameterCollection` object with the specified lambda
    /// value of weiht decay.
    pub fn with_weight_decay(lambda: f32) -> Self {
        let pc = ParameterCollection::new();
        unsafe {
            dn::DN_SetWeightDecay(pc.inner, lambda);
        }
        pc
    }


    /// Add parameters with custom initializer to parameter collection and
    /// returns `Parameter` object. There are many variants of initializer.
    /// 
    /// 
    /// # Panics
    pub fn add_param(&self, dim: &[u32], init: ParamInit, name: &str) -> Parameter {
        let c_name = CString::new(name).unwrap();
        let name_c_ptr = c_name.as_ptr();

        let d:Vec<i64> = dim.iter().map(|d| *d as i64).collect();
        unsafe{
            let dim_ptr = dn::DN_NewDimFromArray(d.as_ptr(), d.len(), 1);
            match init {
                ParamInit::Const(c) => {
                    let raw_param_init = dn::DN_NewParameterInitConst(c);
                    let param = Parameter{inner: dn::DN_AddParametersToCollectionConst(
                        self.inner,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitConst(raw_param_init);
                    param
                },
                ParamInit::Glorot(is_lookup, gain) => {
                    let raw_param_init = dn::DN_NewParameterInitGlorot(is_lookup, gain);
                    let param = Parameter{inner: dn::DN_AddParametersToCollectionGlorot(
                        self.inner,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitGlorot(raw_param_init);
                    param
                }
                ParamInit::Identity => {
                    let raw_param_init = dn::DN_NewParameterInitIdentity();
                    let param = Parameter{inner: dn::DN_AddParametersToCollectionIdentity(
                        self.inner,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitIdentity(raw_param_init);
                    param
                }
                ParamInit::Normal(mean, variance) => {
                    let raw_param_init = dn::DN_NewParameterInitNormal(mean, variance);
                    let param = Parameter{inner: dn::DN_AddParametersToCollectionNormal(
                        self.inner,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitNormal(raw_param_init);
                    param
                }
                ParamInit::Saxe(gain) => {
                    let raw_param_init = dn::DN_NewParameterInitSaxe(gain);
                    let param = Parameter{inner: dn::DN_AddParametersToCollectionSaxe(
                        self.inner,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitSaxe(raw_param_init);
                    param
                }
                ParamInit::Uniform(l, r) => {
                    let raw_param_init = dn::DN_NewParameterInitUniform(l, r);
                    let param = Parameter{inner: dn::DN_AddParametersToCollectionUniform(
                        self.inner,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitUniform(raw_param_init);
                    param
                }
            }
        }
    }

    /// Add parameters with default initializer to parameter collection and
    /// returns `Parameter` object.
    /// 
    ///  Glorot initialization with `gain=1.0` is used in the method.
    pub fn add_default_param(&self, dim: &[u32]) -> Parameter {
        let init = param_init::ParamInit::Glorot(false, 1.0);
        self.add_param(dim, init, "")
    }


    /// Add a lookup parameter to the ParameterCollection with a given
    /// initializer. 
    /// 
    /// The first element of `dim` is the lookup dimension (number of records
    /// in the lookup table).
    /// 
    /// ## Exemple
    /// ```
    /// let vocab_size:u32 = 10000;
    /// let emb_dim:u32 = 200;
    /// 
    /// let pc = dy::ParameterCollection::new();
    /// E = pc.add_lookup_param(&[vocab_size, emb_dim], dy::ParamInit::Glorot(true, 1.0), "E");
    /// ```
    pub fn add_lookup_param(&self, dim: &[u32], init:ParamInit, name: &str) -> LookupParameter {
        if dim.len() < 2 {
            panic!("The length of dimension must be no less than 2, but the provided dimension is {}.", dim.len());
        }
        let c_name = CString::new(name).unwrap();
        let name_c_ptr = c_name.as_ptr();
        let lookup_dim = dim[0];
        let d:Vec<i64> = dim[1..].iter().map(|d| *d as i64).collect();
        unsafe{
            let dim_ptr = dn::DN_NewDimFromArray(d.as_ptr(), d.len(), 1);
            match init {
                ParamInit::Const(c) => {
                    let raw_param_init = dn::DN_NewParameterInitConst(c);
                    let param = LookupParameter{inner: dn::DN_AddLookupParametersToCollectionConst(
                        self.inner,
                        lookup_dim,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitConst(raw_param_init);
                    param
                },
                ParamInit::Glorot(is_lookup, gain) => {
                    let raw_param_init = dn::DN_NewParameterInitGlorot(is_lookup, gain);
                    let param = LookupParameter{inner: dn::DN_AddLookupParametersToCollectionGlorot(
                        self.inner,
                        lookup_dim,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitGlorot(raw_param_init);
                    param
                }
                ParamInit::Identity => {
                    let raw_param_init = dn::DN_NewParameterInitIdentity();
                    let param = LookupParameter{inner: dn::DN_AddLookupParametersToCollectionIdentity(
                        self.inner,
                        lookup_dim,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitIdentity(raw_param_init);
                    param
                }
                ParamInit::Normal(mean, variance) => {
                    let raw_param_init = dn::DN_NewParameterInitNormal(mean, variance);
                    let param = LookupParameter{inner: dn::DN_AddLookupParametersToCollectionNormal(
                        self.inner,
                        lookup_dim,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitNormal(raw_param_init);
                    param
                }
                ParamInit::Saxe(gain) => {
                    let raw_param_init = dn::DN_NewParameterInitSaxe(gain);
                    let param = LookupParameter{inner: dn::DN_AddLookupParametersToCollectionSaxe(
                        self.inner,
                        lookup_dim,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitSaxe(raw_param_init);
                    param
                }
                ParamInit::Uniform(l, r) => {
                    let raw_param_init = dn::DN_NewParameterInitUniform(l, r);
                    let param = LookupParameter{inner: dn::DN_AddLookupParametersToCollectionUniform(
                        self.inner,
                        lookup_dim,
                        dim_ptr,
                        raw_param_init,
                        name_c_ptr
                    )};
                    dn::DN_DeleteDim(dim_ptr);
                    dn::DN_DeleteParameterInitUniform(raw_param_init);
                    param
                }
            }
        }
    }


    /// Add lookup parameters with default initializer to parameter collection
    /// and returns `Parameter` object.
    /// 
    ///  Glorot initialization with `gain=1.0` is used in the method.
    pub fn add_default_lookup_param(&self, dim:&[u32]) -> LookupParameter {
        let init = param_init::ParamInit::Glorot(true, 1.0);
        self.add_lookup_param(dim, init, "")
    }

    /// Get the full name of this collection.
    pub fn name(&self) -> String {
        unsafe {
            let cstr_ptr = dn::DN_GetParameterCollectionFullName(self.inner);
            CStr::from_ptr(cstr_ptr).to_string_lossy().into_owned()
        }
    }

    /// Get the weight decay lambda value.
    pub fn weight_decay_lambda(&self) -> f32 {
        unsafe {
            dn::DN_GetWeightDecayLambda(self.inner)
        }
    }

    /// Get the l2 norm of the gradient.
    /// 
    /// Use this to look for gradient vanishing/exploding.
    pub fn gradient_l2_norm(&self) -> f32 {
        unsafe {
            dn::DN_GradientL2Norm(self.inner)
        }
    }

    /// Get the number of parameters in the `ParameterCollection`.
    pub fn size(&self) -> usize {
        unsafe {
            dn::DN_ParameterCollectionSize(self.inner)
        }
    }
}



////////////////////////////////////////////////////////////////////////////////
// Expression
////////////////////////////////////////////////////////////////////////////////

/// Expressions are the building block of a Dynet computation graph.
/// 
/// They are the main data types being manipulated in a DyNet program. Each
/// expression represents a sub-computation in a computation graph.
pub struct Expression {
    inner: *mut dn::DN_Expression
}

impl_drop!(Expression, DN_DeleteExpression);

impl<'a> Add for &'a Expression {
    type Output = Expression;
    fn add(self, rhs: Self) -> Expression {
        unsafe {
            Expression{inner: dn::DN_Add(self.inner, rhs.inner)}
        }
    }
}

impl<'a> Mul for &'a Expression {
    type Output = Expression;
    fn mul(self, rhs: Self) -> Expression {
        unsafe {
            Expression{inner: dn::DN_Multiply(self.inner, rhs.inner)}
        }
    }
}

impl Expression {
    pub fn print(&self) {
        unsafe {
            let tensor = dn::DN_GetExprValue(self.inner);
            dn::DN_PrintTensor(tensor);
        }
    }
}

mod dim;
pub use dim::{Dim};

mod param_init;
pub use param_init::{ParamInit};

mod graph;
pub use graph::{ComputationGraph, get_current_graph_id};

mod init;
pub use init::{initialize, reset_rand_seed};

pub mod trainer;

pub mod ops;
