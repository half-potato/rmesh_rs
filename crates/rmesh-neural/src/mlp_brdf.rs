//! MLPBRDF: Neural specular BRDF model.
//!
//! Predicts specular color and diffuse/specular blend factor (lambda) from
//! half-vector, view direction, per-pixel env features, and roughness.

use burn::nn;
use burn::prelude::*;

use crate::ish::ish_basis;

/// Weight data for a single linear layer (loaded from rmesh-data).
pub struct LayerData {
    pub in_dim: usize,
    pub out_dim: usize,
    pub has_bias: bool,
    pub weights: Vec<f32>, // [out_dim * in_dim] row-major
    pub bias: Vec<f32>,    // [out_dim]
}

/// MLPBRDF model: ISH encoding + MLP → (spec_color, lambda).
#[derive(Module, Debug)]
pub struct MlpBrdf<B: Backend> {
    layers: Vec<nn::Linear<B>>,
    bias_offset: f32,
}

impl<B: Backend> MlpBrdf<B> {
    /// Construct from pre-trained layer weights.
    pub fn load(layer_data: &[LayerData], bias_offset: f32, device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(layer_data.len());
        for ld in layer_data {
            let weight_tensor =
                Tensor::<B, 1>::from_floats(&ld.weights[..], device).reshape([ld.out_dim, ld.in_dim]);
            let config = nn::LinearConfig::new(ld.in_dim, ld.out_dim).with_bias(ld.has_bias);
            let mut linear = config.init(device);

            linear.weight = burn::module::Param::from_tensor(weight_tensor);

            if ld.has_bias && !ld.bias.is_empty() {
                let bias_tensor = Tensor::<B, 1>::from_floats(&ld.bias[..], device);
                linear.bias = Some(burn::module::Param::from_tensor(bias_tensor));
            }

            layers.push(linear);
        }
        Self {
            layers,
            bias_offset,
        }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `half_vec` - Half-vectors [N, 3]
    /// * `view_dir` - View directions [N, 3]
    /// * `env_features` - Per-pixel features [N, 4]
    /// * `roughness` - Per-pixel roughness [N, 1]
    ///
    /// # Returns
    /// (spec_color [N, 3], lambda [N, 1])
    pub fn forward(
        &self,
        half_vec: Tensor<B, 2>,
        view_dir: Tensor<B, 2>,
        env_features: Tensor<B, 2>,
        roughness: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Compute kappa = 1 / (roughness + 1e-3)
        let kappa = (roughness.clone() + 1e-3).recip();

        // ISH encoding for half vector and view direction
        let ish_half = ish_basis(half_vec.clone(), kappa.clone()); // [N, 18]
        let ish_view = ish_basis(view_dir.clone(), kappa); // [N, 18]

        // Concatenate input: [env_features(4), ish_half(18), half_vec(3), ish_view(18), view_dir(3)]
        let input = Tensor::cat(
            vec![env_features, ish_half, half_vec, ish_view, view_dir],
            1,
        ); // [N, 46]

        // Forward through MLP with ReLU
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            // ReLU on all but last layer
            if i < self.layers.len() - 1 {
                x = burn::tensor::activation::relu(x);
            }
        }

        // Output: sigmoid(x[:, :3] + bias) = spec_color, sigmoid(x[:, 3:]) = lambda
        let n = x.dims()[0];
        let spec_raw = x.clone().slice([0..n, 0..3]) + self.bias_offset;
        let lam_raw = x.slice([0..n, 3..4]);

        let spec_color = burn::tensor::activation::sigmoid(spec_raw);
        let lambda = burn::tensor::activation::sigmoid(lam_raw);

        (spec_color, lambda)
    }
}
