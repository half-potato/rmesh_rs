//! Fast shader iteration: bake WGSL in at compile time, but optionally reload
//! it from the source file at runtime so you can edit a `.wgsl` and just rerun
//! the already-built binary — no `cargo` rebuild/relink.
//!
//! Replace
//! ```ignore
//! const FOO_WGSL: &str = include_str!("wgsl/foo.wgsl");
//! ```
//! with
//! ```ignore
//! static FOO_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/foo.wgsl");
//! ```
//! [`HotShader`] derefs to `str`, so method calls (`FOO_WGSL.replace(..)`) and
//! reference uses (`&FOO_WGSL`, `&*FOO_WGSL`) keep working as-is. The one place
//! that needs a tweak is `wgpu::ShaderSource::Wgsl(FOO_WGSL.into())` — because
//! `Into::into` takes `self` by value it does not deref-coerce, so write
//! `FOO_WGSL.as_str().into()` instead.
//!
//! At runtime, when the `RMESH_HOT_SHADERS` env var is set, the first use of
//! each shader reads its source file from disk (the `CARGO_MANIFEST_DIR/src/...`
//! path baked by the macro). On any read error it falls back to the compiled-in
//! copy. The result is cached for the rest of the process, so reloading is a
//! matter of editing the file and restarting the binary. Release builds that
//! never set the env var pay nothing beyond a single atomic load per shader.

use std::borrow::Cow;
use std::sync::OnceLock;

/// A WGSL shader source that is compiled in but can be transparently reloaded
/// from disk at runtime. Construct via the [`hot_shader!`](crate::hot_shader)
/// macro and store in a `static`. Derefs to `str`.
pub struct HotShader {
    /// Absolute path to the source `.wgsl`, baked from `CARGO_MANIFEST_DIR`.
    path: &'static str,
    /// Compiled-in fallback (the `include_str!` contents).
    baked: &'static str,
    resolved: OnceLock<Cow<'static, str>>,
}

impl HotShader {
    /// Used by the [`hot_shader!`](crate::hot_shader) macro; prefer that.
    pub const fn new(path: &'static str, baked: &'static str) -> Self {
        Self {
            path,
            baked,
            resolved: OnceLock::new(),
        }
    }

    fn resolve(&self) -> &str {
        self.resolved.get_or_init(|| {
            if std::env::var_os("RMESH_HOT_SHADERS").is_some() {
                match std::fs::read_to_string(self.path) {
                    Ok(src) => {
                        eprintln!("[hot-shader] reloaded {}", self.path);
                        return Cow::Owned(src);
                    }
                    Err(e) => {
                        eprintln!(
                            "[hot-shader] failed to read {} ({e}); using baked copy",
                            self.path
                        );
                    }
                }
            }
            Cow::Borrowed(self.baked)
        })
    }

    /// The shader source as a string slice.
    pub fn as_str(&self) -> &str {
        self.resolve()
    }
}

impl std::ops::Deref for HotShader {
    type Target = str;
    fn deref(&self) -> &str {
        self.resolve()
    }
}

impl std::fmt::Display for HotShader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.resolve())
    }
}

/// Define a hot-reloadable shader source. Expands to a [`HotShader`] holding the
/// `include_str!`-baked copy plus the on-disk path (`CARGO_MANIFEST_DIR/src/` +
/// the given relative path) for runtime reload. Use in a `static`:
///
/// ```ignore
/// static FOO_WGSL: rmesh_util::HotShader = rmesh_util::hot_shader!("wgsl/foo.wgsl");
/// ```
///
/// The relative path is resolved the same way `include_str!` is — relative to
/// the source file — with a `src/` prefix for the runtime path, matching the
/// convention that shader consts live in files directly under `src/`.
#[macro_export]
macro_rules! hot_shader {
    ($rel:literal) => {
        $crate::HotShader::new(
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/", $rel),
            include_str!($rel),
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_back_to_baked_when_not_opted_in() {
        // No env var, and a path that does not exist: must use the baked copy.
        let s = HotShader::new("/nonexistent/does/not/exist.wgsl", "BAKED CONTENTS");
        assert_eq!(&*s, "BAKED CONTENTS");
        assert_eq!(s.as_str(), "BAKED CONTENTS");
    }

    #[test]
    fn reloads_from_disk_when_opted_in() {
        // Write a file, point a HotShader at it with the env var set, and
        // confirm the deref returns the on-disk contents, not the baked copy.
        let mut path = std::env::temp_dir();
        path.push(format!("rmesh_hot_shader_test_{}.wgsl", std::process::id()));
        std::fs::write(&path, "ON DISK").expect("write temp shader");

        let s = HotShader::new(
            Box::leak(path.to_string_lossy().into_owned().into_boxed_str()),
            "BAKED",
        );
        // SAFETY: single-threaded within this test; we set then unset.
        unsafe { std::env::set_var("RMESH_HOT_SHADERS", "1") };
        let got = s.as_str().to_string();
        unsafe { std::env::remove_var("RMESH_HOT_SHADERS") };
        let _ = std::fs::remove_file(&path);

        assert_eq!(got, "ON DISK", "should have reloaded from disk");
    }
}
