//! Neural material evaluation for PBR relighting.
//!
//! Provides MLPBRDF (specular BRDF model) and RetroHead (retroreflectivity)
//! using the burn framework for inference.

pub mod ish;
pub mod mlp_brdf;
pub mod retro;

use burn::prelude::*;
pub use mlp_brdf::{LayerData, MlpBrdf};
pub use retro::RetroHead;

/// Combined neural material models.
#[derive(Module, Debug)]
pub struct NeuralMaterials<B: Backend> {
    pub brdf: MlpBrdf<B>,
    pub retro: RetroHead<B>,
}

impl<B: Backend> NeuralMaterials<B> {
    /// Load from weight data (as parsed by rmesh-data PbrData).
    pub fn load(
        brdf_layers: &[LayerData],
        brdf_bias: f32,
        retro_weights: &[f32],
        retro_bias: &[f32],
        device: &B::Device,
    ) -> Self {
        Self {
            brdf: MlpBrdf::load(brdf_layers, brdf_bias, device),
            retro: RetroHead::load(retro_weights, retro_bias, device),
        }
    }

    /// Evaluate neural materials for a batch of visible pixels.
    ///
    /// # Arguments
    /// * `half_vecs` - Half-vectors [N, 3]
    /// * `view_dirs` - View directions (camera → surface) [N, 3]
    /// * `env_features` - Per-pixel specular features [N, 4]
    /// * `roughness` - Per-pixel roughness [N, 1]
    ///
    /// # Returns
    /// (spec_color [N, 3], lambda [N, 1], retro [N, 1])
    pub fn evaluate(
        &self,
        half_vecs: Tensor<B, 2>,
        view_dirs: Tensor<B, 2>,
        env_features: Tensor<B, 2>,
        roughness: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let (spec_color, lambda) =
            self.brdf
                .forward(half_vecs, view_dirs, env_features.clone(), roughness);
        let retro = self.retro.forward(env_features);
        (spec_color, lambda, retro)
    }
}
