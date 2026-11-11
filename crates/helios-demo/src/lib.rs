//! Helios demo: shared application code plus the browser entry point.

pub mod app;
pub mod app_editor;
pub mod editor;
pub mod mesh_store;
pub mod motion;
pub mod scene_bake;
#[cfg(not(target_arch = "wasm32"))]
pub mod native_editor;
pub mod scenes;
pub mod tour;

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
mod web_bake;
#[cfg(target_arch = "wasm32")]
mod web_editor;
