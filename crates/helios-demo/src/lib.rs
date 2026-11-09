//! Helios demo: shared application code plus the browser entry point.

pub mod app;
pub mod mesh_store;
pub mod scenes;
pub mod tour;

#[cfg(target_arch = "wasm32")]
mod web;
