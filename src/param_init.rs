/// Initializers for parameters.
/// 
/// 
#[derive(Debug, Clone, Copy)]
pub enum ParamInit {
    /// Initialize parameters with a constant value.
    /// 
    /// *Associated values:* `(c: f32)` where `c` is short for 'constant value' 
    Const(f32),

    /// Initialize with the methods described in [Glorot, 2010](http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf?hc_location=ufi)
    /// 
    /// Important note : The underlying distribution is uniform (not gaussian)
    /// 
    /// *Note:* This is also known as **Xavier initialization**
    /// 
    /// *Associated values:* `(is_lookup: bool, gain: f32)` where `is_lookup`  identify the parameter as a LookupParameter and 
    /// 
    /// Choose `is_lookup=false` and `gain=1.0` for the default values.
    Glorot(bool, f32),

    /// Initialize as the identity
    Identity,

    /// Initialize parameters with samples from a normal distribution
    /// 
    /// *Associated values:* `(mean: f32, variance: f32)`
    /// 
    /// Choose `mean=0.0` and `variance=1.0` for the default values.
    Normal(f32, f32),

    /// Initializes according to [Saxe et al., 2014](https://arxiv.org/abs/1312.6120)
    /// 
    /// Initializes as a random orthogonal matrix (unimplemented for GPU)
    /// 
    /// *Associated values:* `(gain: f32)`
    /// 
    /// Choose `gain=1.0` for the default value.
    Saxe(f32),

    /// Initialize parameters with samples from a uniform distribution
    /// 
    /// *Associated values:* `(l: f32, r: f32)` where `l` is short for 'lower bound of the interval' and `r` is short for 'Upper bound of the interval'
    Uniform(f32, f32),
}