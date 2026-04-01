//! RetroHead: predicts retroreflectivity from per-pixel env features.
//!
//! Single linear layer (4 → 1) + sigmoid.

use burn::nn;
use burn::prelude::*;

/// RetroHead model.
#[derive(Module, Debug)]
pub struct RetroHead<B: Backend> {
    linear: nn::Linear<B>,
}

impl<B: Backend> RetroHead<B> {
    /// Construct from pre-trained weights.
    ///
    /// * `weights` - [4] f32
    /// * `bias` - [1] f32
    pub fn load(weights: &[f32], bias: &[f32], device: &B::Device) -> Self {
        let config = nn::LinearConfig::new(4, 1).with_bias(!bias.is_empty());
        let mut linear = config.init(device);

        let w = Tensor::<B, 1>::from_floats(weights, device).reshape([1, 4]);
        linear.weight = burn::module::Param::from_tensor(w);

        if !bias.is_empty() {
            let b = Tensor::<B, 1>::from_floats(bias, device);
            linear.bias = Some(burn::module::Param::from_tensor(b));
        }

        Self { linear }
    }

    /// Forward pass.
    ///
    /// * `env_features` - [N, 4]
    ///
    /// Returns retroreflectivity [N, 1] in [0, 1].
    pub fn forward(&self, env_features: Tensor<B, 2>) -> Tensor<B, 2> {
        burn::tensor::activation::sigmoid(self.linear.forward(env_features))
    }
}
