//! Isotropic Spherical Harmonics (ISH) basis evaluation.
//!
//! Evaluates basis functions at degrees [0, 1, 2, 4] (degree 3 skipped),
//! giving 1 + 3 + 5 + 9 = 18 coefficients per direction vector.
//! Each coefficient is modulated by Al(l, kappa) = exp(-l*(l+1)/2/kappa).

use burn::prelude::*;

/// Total number of ISH coefficients: deg0(1) + deg1(3) + deg2(5) + deg4(9) = 18.
pub const ISH_DIM: usize = 18;

/// Evaluate ISH basis for a batch of direction vectors.
///
/// # Arguments
/// * `dirs` - Unit direction vectors [N, 3]
/// * `kappa` - Sharpness parameter [N, 1], typically 1/(roughness + 1e-3)
///
/// # Returns
/// ISH basis values [N, 18]
pub fn ish_basis<B: Backend>(dirs: Tensor<B, 2>, kappa: Tensor<B, 2>) -> Tensor<B, 2> {
    let device = dirs.device();
    let n = dirs.dims()[0];

    // Extract x, y, z components
    let x = dirs.clone().slice([0..n, 0..1]).squeeze(1); // [N]
    let y = dirs.clone().slice([0..n, 1..2]).squeeze(1); // [N]
    let z = dirs.slice([0..n, 2..3]).squeeze(1); // [N]

    let xx = x.clone() * x.clone();
    let yy = y.clone() * y.clone();
    let zz = z.clone() * z.clone();
    let kappa_flat = kappa.squeeze(1); // [N]

    // Al(l, kappa) = exp(-l*(l+1)/2 / (kappa + 1e-8))
    let eps = 1e-8;
    let kappa_safe = kappa_flat.clone() + eps;

    let al0 = Tensor::<B, 1>::ones([n], &device); // exp(0) = 1
    let al1 = (Tensor::<B, 1>::ones([n], &device) * (-1.0) / kappa_safe.clone()).exp();
    let al2 = (Tensor::<B, 1>::ones([n], &device) * (-3.0) / kappa_safe.clone()).exp();
    let al4 = (Tensor::<B, 1>::ones([n], &device) * (-10.0) / kappa_safe).exp();

    // Degree 0: 1 coefficient
    let c00 = al0 * 0.28209479177387814;

    // Degree 1: 3 coefficients
    let c1m1 = al1.clone() * (-0.488603) * x.clone();
    let c10 = al1.clone() * 0.488603 * z.clone();
    let c1p1 = al1 * (-0.488603) * y.clone();

    // Degree 2: 5 coefficients
    let c2m2 = al2.clone() * 1.092548 * y.clone() * x.clone();
    let c2m1 = al2.clone() * (-1.092548) * y.clone() * z.clone();
    let c20 = al2.clone() * 0.315392 * (zz.clone() * 3.0 - 1.0);
    let c2p1 = al2.clone() * (-1.092548) * x.clone() * y.clone();
    let c2p2 = al2 * 0.546274 * (xx.clone() - yy.clone());

    // Degree 4: 9 coefficients
    let x4 = xx.clone() * xx.clone();
    let y4 = yy.clone() * yy.clone();
    let z4 = zz.clone() * zz.clone();

    let c4m4 = al4.clone() * 2.50334 * x.clone() * y.clone() * (xx.clone() - yy.clone());
    let c4m3 = al4.clone() * (-1.77013) * y.clone() * z.clone() * (xx.clone() * (-3.0) + yy.clone());
    let c4m2 = al4.clone() * 0.946175 * x.clone() * y.clone() * (zz.clone() * 7.0 - 1.0);
    let c4m1 = al4.clone() * 0.669047 * y.clone() * z.clone() * (zz.clone() * 7.0 - 3.0);
    let c40 = al4.clone() * (z4 * 3.70251 - zz.clone() * 3.17358 + 0.317358);
    let c4p1 = al4.clone() * 0.669047 * x.clone() * z.clone() * (zz.clone() * 7.0 - 3.0);
    let c4p2 = al4.clone() * (xx.clone() - yy.clone()) * (zz * 7.0 - 1.0) * 0.473087;
    let c4p3 = al4.clone() * 1.77013 * x.clone() * z * (xx.clone() - yy.clone() * 3.0);
    let c4p4 = al4 * (x4 - xx * yy * 6.0 + y4) * 0.625836;

    // Stack all 18 coefficients: [N, 18]
    Tensor::stack(
        vec![
            c00, c1m1, c10, c1p1, c2m2, c2m1, c20, c2p1, c2p2, c4m4, c4m3, c4m2, c4m1, c40,
            c4p1, c4p2, c4p3, c4p4,
        ],
        1,
    )
}
