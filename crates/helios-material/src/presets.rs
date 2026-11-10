//! Example graphs (the gallery scene's and the editor's starting points).

use crate::graph::{GraphBuilder, MaterialGraph, Op};

/// Preset ids and labels, for editors.
pub const PRESETS: [(&str, &str); 5] = [
    ("blank", "Blank (output only)"),
    ("lava", "Animated lava"),
    ("wet_stone", "Wet stone with puddles"),
    ("stylized_water", "Stylised water"),
    ("brick_pavement", "Brick pavement (displaced)"),
];

/// The preset graph `id`.
pub fn preset(id: &str) -> Option<MaterialGraph> {
    match id {
        "blank" => Some(blank()),
        "lava" => Some(lava()),
        "wet_stone" => Some(wet_stone()),
        "stylized_water" => Some(stylized_water()),
        "brick_pavement" => Some(brick_pavement()),
        _ => None,
    }
}

/// Only an output: the material keeps its own values.
pub fn blank() -> MaterialGraph {
    let mut g = GraphBuilder::new("Blank");
    g.node(Op::Output, [0.0, 0.0]);
    g.build()
}

/// Molten rock: fractal noise scrolling across the surface (world xz, slowly churning in
/// time) splits a dark, rough crust from glowing, smoother melt between its plates.
pub fn lava() -> MaterialGraph {
    let mut g = GraphBuilder::new("Lava");
    let position = g.node(Op::WorldPosition, [-1150.0, -40.0]);
    let xz = g.node(Op::ComponentMask { channels: "xz".into() }, [-960.0, -40.0]);
    let pan = g.node(Op::Panner { speed: [0.04, 0.015] }, [-780.0, -40.0]);
    let time = g.node(Op::Time, [-960.0, 80.0]);
    let churn = g.node(Op::Multiply, [-780.0, 80.0]);
    let point = g.node(Op::Append, [-600.0, 0.0]);
    let noise = g.node(Op::Noise { scale: 0.9, octaves: 5 }, [-430.0, 0.0]);
    let crust = g.node(Op::SmoothStep, [-260.0, 0.0]);
    let molten = g.node(Op::OneMinus, [-90.0, 120.0]);
    let veins = g.node(Op::Power, [70.0, 120.0]);
    let glow_color = g.node(Op::VectorParameter { name: "Glow Color".into(), default: [1.0, 0.32, 0.04, 1.0] }, [-90.0, 240.0]);
    let glow = g.node(Op::ScalarParameter { name: "Glow Intensity".into(), default: 14.0, min: 0.0, max: 40.0 }, [-90.0, 340.0]);
    let tinted = g.node(Op::Multiply, [70.0, 260.0]);
    let emissive = g.node(Op::Multiply, [240.0, 180.0]);
    let melt = g.node(Op::Constant3 { value: [0.6, 0.18, 0.03] }, [-90.0, -200.0]);
    let rock = g.node(Op::Constant3 { value: [0.035, 0.03, 0.028] }, [-90.0, -110.0]);
    let base = g.node(Op::Lerp, [240.0, -150.0]);
    let roughness = g.node(Op::Lerp, [240.0, -20.0]);
    let output = g.node(Op::Output, [460.0, 0.0]);
    g.link(position, "Out", xz, "In")
        .link(xz, "Out", pan, "UV")
        .link(time, "Out", churn, "A")
        .value(churn, "B", 0.15)
        .link(pan, "Out", point, "A")
        .link(churn, "Out", point, "B")
        .link(point, "Out", noise, "Position")
        .value(crust, "Min", 0.42)
        .value(crust, "Max", 0.58)
        .link(noise, "Out", crust, "X")
        .link(crust, "Out", molten, "In")
        .link(molten, "Out", veins, "Base")
        .value(veins, "Exponent", 1.5)
        .link(glow_color, "Out", tinted, "A")
        .link(glow, "Out", tinted, "B")
        .link(tinted, "Out", emissive, "A")
        .link(veins, "Out", emissive, "B")
        .link(melt, "Out", base, "A")
        .link(rock, "Out", base, "B")
        .link(crust, "Out", base, "Alpha")
        .value(roughness, "A", 0.4)
        .value(roughness, "B", 0.92)
        .link(crust, "Out", roughness, "Alpha")
        .link(base, "Out", output, "BaseColor")
        .link(roughness, "Out", output, "Roughness")
        .link(emissive, "Out", output, "Emissive");
    g.build()
}

/// Rain-soaked stone: the texture of parameter "Stone" projected triplanar, darkened and made
/// glossy by "Wetness", with mirror-flat puddles where noise dips below "Puddle Level" on
/// upward-facing surfaces.
pub fn wet_stone() -> MaterialGraph {
    let mut g = GraphBuilder::new("Wet stone");
    let stone = g.node(Op::Triplanar { parameter: "Stone".into(), scale: 0.6 }, [-700.0, -160.0]);
    let position = g.node(Op::WorldPosition, [-1060.0, 120.0]);
    let noise = g.node(Op::Noise { scale: 0.7, octaves: 4 }, [-880.0, 120.0]);
    let level = g.node(Op::ScalarParameter { name: "Puddle Level".into(), default: 0.45, min: 0.0, max: 1.0 }, [-880.0, 230.0]);
    let level_top = g.node(Op::Add, [-700.0, 260.0]);
    let dry = g.node(Op::SmoothStep, [-520.0, 160.0]);
    let pooled = g.node(Op::OneMinus, [-430.0, 160.0]);
    let normal = g.node(Op::VertexNormal, [-880.0, 360.0]);
    let up = g.node(Op::ComponentMask { channels: "y".into() }, [-700.0, 360.0]);
    let flat = g.node(Op::SmoothStep, [-520.0, 340.0]);
    let puddle = g.node(Op::Multiply, [-340.0, 240.0]);
    let wetness = g.node(Op::ScalarParameter { name: "Wetness".into(), default: 0.6, min: 0.0, max: 1.0 }, [-340.0, 360.0]);
    let wet = g.node(Op::Max, [-160.0, 300.0]);
    let darken = g.node(Op::Lerp, [20.0, 160.0]);
    let base = g.node(Op::Multiply, [200.0, -160.0]);
    let damp = g.node(Op::Lerp, [20.0, 0.0]);
    let roughness = g.node(Op::Lerp, [200.0, 40.0]);
    let up_normal = g.node(Op::Constant3 { value: [0.0, 0.0, 1.0] }, [-160.0, -60.0]);
    let blend = g.node(Op::Lerp, [20.0, -100.0]);
    let tangent = g.node(Op::Normalize, [200.0, -60.0]);
    let output = g.node(Op::Output, [420.0, 0.0]);
    g.link(position, "Out", noise, "Position")
        .link(level, "Out", level_top, "A")
        .value(level_top, "B", 0.04)
        .link(level, "Out", dry, "Min")
        .link(level_top, "Out", dry, "Max")
        .link(noise, "Out", dry, "X")
        .link(dry, "Out", pooled, "In")
        .link(normal, "Out", up, "In")
        .value(flat, "Min", 0.8)
        .value(flat, "Max", 0.97)
        .link(up, "Out", flat, "X")
        .link(pooled, "Out", puddle, "A")
        .link(flat, "Out", puddle, "B")
        .link(wetness, "Out", wet, "A")
        .link(puddle, "Out", wet, "B")
        .value(darken, "A", 1.0)
        .value(darken, "B", 0.35)
        .link(wet, "Out", darken, "Alpha")
        .link(stone, "RGB", base, "A")
        .link(darken, "Out", base, "B")
        .link(stone, "Roughness", damp, "A")
        .value(damp, "B", 0.12)
        .link(wetness, "Out", damp, "Alpha")
        .link(damp, "Out", roughness, "A")
        .value(roughness, "B", 0.02)
        .link(puddle, "Out", roughness, "Alpha")
        .link(stone, "TangentNormal", blend, "A")
        .link(up_normal, "Out", blend, "B")
        .link(puddle, "Out", blend, "Alpha")
        .link(blend, "Out", tangent, "In")
        .link(base, "Out", output, "BaseColor")
        .link(roughness, "Out", output, "Roughness")
        .link(tangent, "Out", output, "Normal");
    g.build()
}

/// Toon water on an opaque surface: two noise layers drifting in different directions make
/// ripples (their difference tilts the normal) and crisp foam bands where they peak; the colour
/// goes from "Shallow Color" facing the camera to "Deep Color" at grazing angles.
pub fn stylized_water() -> MaterialGraph {
    let mut g = GraphBuilder::new("Stylised water");
    let position = g.node(Op::WorldPosition, [-1200.0, 0.0]);
    let xz = g.node(Op::ComponentMask { channels: "xz".into() }, [-1030.0, 0.0]);
    let coarse = g.node(Op::Multiply, [-860.0, -80.0]);
    let fine = g.node(Op::Multiply, [-860.0, 80.0]);
    let pan_a = g.node(Op::Panner { speed: [0.08, 0.03] }, [-690.0, -80.0]);
    let pan_b = g.node(Op::Panner { speed: [-0.05, 0.07] }, [-690.0, 80.0]);
    let noise_a = g.node(Op::Noise { scale: 1.0, octaves: 3 }, [-520.0, -80.0]);
    let noise_b = g.node(Op::Noise { scale: 1.0, octaves: 3 }, [-520.0, 80.0]);
    let waves = g.node(Op::Add, [-350.0, 0.0]);
    let foam = g.node(Op::SmoothStep, [-180.0, 0.0]);
    let fresnel = g.node(Op::Fresnel { exponent: 3.0, base_reflect: 0.02 }, [-350.0, -220.0]);
    let shallow = g.node(Op::VectorParameter { name: "Shallow Color".into(), default: [0.05, 0.42, 0.52, 1.0] }, [-350.0, -400.0]);
    let deep = g.node(Op::VectorParameter { name: "Deep Color".into(), default: [0.01, 0.07, 0.18, 1.0] }, [-350.0, -310.0]);
    let water = g.node(Op::Lerp, [-180.0, -300.0]);
    let white = g.node(Op::Constant4 { value: [0.92, 0.96, 1.0, 1.0] }, [-180.0, -160.0]);
    let base = g.node(Op::Lerp, [0.0, -200.0]);
    let roughness = g.node(Op::Lerp, [0.0, 0.0]);
    let tilt = g.node(Op::Subtract, [-350.0, 200.0]);
    let slope = g.node(Op::Multiply, [-180.0, 200.0]);
    let slope_xy = g.node(Op::Append, [0.0, 200.0]);
    let tangent = g.node(Op::Append, [160.0, 200.0]);
    let normal = g.node(Op::Normalize, [320.0, 200.0]);
    let specular = g.node(Op::Constant { value: 0.6 }, [160.0, 100.0]);
    let output = g.node(Op::Output, [520.0, 0.0]);
    g.link(position, "Out", xz, "In")
        .link(xz, "Out", coarse, "A")
        .value(coarse, "B", 0.6)
        .link(xz, "Out", fine, "A")
        .value(fine, "B", 1.4)
        .link(coarse, "Out", pan_a, "UV")
        .link(fine, "Out", pan_b, "UV")
        .link(pan_a, "Out", noise_a, "Position")
        .link(pan_b, "Out", noise_b, "Position")
        .link(noise_a, "Out", waves, "A")
        .link(noise_b, "Out", waves, "B")
        .value(foam, "Min", 1.18)
        .value(foam, "Max", 1.24)
        .link(waves, "Out", foam, "X")
        .link(shallow, "Out", water, "A")
        .link(deep, "Out", water, "B")
        .link(fresnel, "Out", water, "Alpha")
        .link(water, "Out", base, "A")
        .link(white, "Out", base, "B")
        .link(foam, "Out", base, "Alpha")
        .value(roughness, "A", 0.06)
        .value(roughness, "B", 0.7)
        .link(foam, "Out", roughness, "Alpha")
        .link(noise_a, "Out", tilt, "A")
        .link(noise_b, "Out", tilt, "B")
        .link(tilt, "Out", slope, "A")
        .value(slope, "B", 0.8)
        .link(slope, "Out", slope_xy, "A")
        .link(tilt, "Out", slope_xy, "B")
        .link(slope_xy, "Out", tangent, "A")
        .value(tangent, "B", 1.0)
        .link(tangent, "Out", normal, "In")
        .link(base, "Out", output, "BaseColor")
        .link(roughness, "Out", output, "Roughness")
        .link(normal, "Out", output, "Normal")
        .link(specular, "Out", output, "Specular");
    g.build()
}

/// A brick pavement in running bond whose Displacement output carves the mortar joints out of
/// the surface (Nanite tessellation): world xz in brick units, every other row shifted by half a
/// brick; the distance to the brick's edge, bevelled by a smoothstep, separates brick from mortar
/// ("Mortar Width"), coarser noise chips the bricks' edges, a per-brick random value (noise at the
/// brick's cell) sets the bricks at slightly different heights ("Brick Variation") and tints them,
/// and fine noise roughens both.
/// Its material needs a displacement magnitude (the gallery uses 3.5 cm, centre 0.9).
pub fn brick_pavement() -> MaterialGraph {
    let mut g = GraphBuilder::new("Brick pavement");
    let position = g.node(Op::WorldPosition, [-1600.0, 0.0]);
    let xz = g.node(Op::ComponentMask { channels: "xz".into() }, [-1430.0, 0.0]);
    let per_brick = g.node(Op::Constant2 { value: [1.0 / 0.22, 1.0 / 0.11] }, [-1430.0, 100.0]);
    let bricks = g.node(Op::Multiply, [-1260.0, 0.0]);
    let row = g.node(Op::ComponentMask { channels: "y".into() }, [-1090.0, 80.0]);
    let row_index = g.node(Op::Floor, [-920.0, 80.0]);
    let half_row = g.node(Op::Multiply, [-750.0, 80.0]);
    let stagger = g.node(Op::Frac, [-580.0, 80.0]);
    let column = g.node(Op::ComponentMask { channels: "x".into() }, [-1090.0, -40.0]);
    let shifted_column = g.node(Op::Add, [-410.0, -40.0]);
    let shifted = g.node(Op::Append, [-240.0, 0.0]);
    let cell = g.node(Op::Floor, [-70.0, 160.0]);
    let local = g.node(Op::Frac, [-70.0, 0.0]);
    let far_side = g.node(Op::OneMinus, [100.0, 60.0]);
    let nearest = g.node(Op::Min, [270.0, 0.0]);
    let brick_size = g.node(Op::Constant2 { value: [0.22, 0.11] }, [270.0, 100.0]);
    let edge_metres = g.node(Op::Multiply, [440.0, 0.0]);
    let edge_x = g.node(Op::ComponentMask { channels: "x".into() }, [610.0, -40.0]);
    let edge_y = g.node(Op::ComponentMask { channels: "y".into() }, [610.0, 40.0]);
    let edge = g.node(Op::Min, [780.0, 0.0]);
    let chips = g.node(Op::Noise { scale: 9.0, octaves: 2 }, [610.0, -140.0]);
    let chip_depth = g.node(Op::Multiply, [780.0, -140.0]);
    let chipped = g.node(Op::Add, [950.0, -60.0]);
    let mortar_width = g.node(Op::ScalarParameter { name: "Mortar Width".into(), default: 0.01, min: 0.0, max: 0.03 }, [610.0, 160.0]);
    let bevel_end = g.node(Op::Add, [780.0, 160.0]);
    let brick = g.node(Op::SmoothStep, [950.0, 60.0]);
    let seed = g.node(Op::Multiply, [100.0, 200.0]);
    let noise = g.node(Op::Noise { scale: 1.0, octaves: 1 }, [270.0, 220.0]);
    let random = g.node(Op::SmoothStep, [440.0, 200.0]);
    let variation = g.node(Op::ScalarParameter { name: "Brick Variation".into(), default: 0.45, min: 0.0, max: 1.0 }, [440.0, 320.0]);
    let sunk = g.node(Op::Multiply, [610.0, 260.0]);
    let top = g.node(Op::Subtract, [780.0, 260.0]);
    let grain = g.node(Op::Noise { scale: 18.0, octaves: 2 }, [610.0, 420.0]);
    let grain_height = g.node(Op::Multiply, [780.0, 420.0]);
    let rough_top = g.node(Op::Add, [950.0, 300.0]);
    let joint = g.node(Op::Add, [950.0, 420.0]);
    let height = g.node(Op::Lerp, [1120.0, 300.0]);
    let brick_color = g.node(Op::VectorParameter { name: "Brick Color".into(), default: [0.42, 0.15, 0.08, 1.0] }, [610.0, -320.0]);
    let brick_rgb = g.node(Op::ComponentMask { channels: "rgb".into() }, [780.0, -320.0]);
    let tint = g.node(Op::Lerp, [780.0, -200.0]);
    let tinted = g.node(Op::Multiply, [950.0, -260.0]);
    let mortar_color = g.node(Op::Constant3 { value: [0.46, 0.44, 0.4] }, [950.0, -380.0]);
    let base = g.node(Op::Lerp, [1120.0, -260.0]);
    let roughness = g.node(Op::Lerp, [1120.0, -100.0]);
    let output = g.node(Op::Output, [1320.0, 0.0]);
    g.link(position, "Out", xz, "In")
        .link(xz, "Out", bricks, "A")
        .link(per_brick, "Out", bricks, "B")
        .link(bricks, "Out", row, "In")
        .link(row, "Out", row_index, "In")
        .link(row_index, "Out", half_row, "A")
        .value(half_row, "B", 0.5)
        .link(half_row, "Out", stagger, "In")
        .link(bricks, "Out", column, "In")
        .link(column, "Out", shifted_column, "A")
        .link(stagger, "Out", shifted_column, "B")
        .link(shifted_column, "Out", shifted, "A")
        .link(row, "Out", shifted, "B")
        .link(shifted, "Out", cell, "In")
        .link(shifted, "Out", local, "In")
        .link(local, "Out", far_side, "In")
        .link(local, "Out", nearest, "A")
        .link(far_side, "Out", nearest, "B")
        .link(nearest, "Out", edge_metres, "A")
        .link(brick_size, "Out", edge_metres, "B")
        .link(edge_metres, "Out", edge_x, "In")
        .link(edge_metres, "Out", edge_y, "In")
        .link(edge_x, "Out", edge, "A")
        .link(edge_y, "Out", edge, "B")
        .link(mortar_width, "Out", bevel_end, "A")
        .value(bevel_end, "B", 0.014)
        .link(mortar_width, "Out", brick, "Min")
        .link(bevel_end, "Out", brick, "Max")
        .link(position, "Out", chips, "Position")
        .link(chips, "Out", chip_depth, "A")
        .value(chip_depth, "B", 0.01)
        .link(edge, "Out", chipped, "A")
        .link(chip_depth, "Out", chipped, "B")
        .link(chipped, "Out", brick, "X")
        .link(cell, "Out", seed, "A")
        .value(seed, "B", 0.613)
        .link(seed, "Out", noise, "Position")
        .value(random, "Min", 0.3)
        .value(random, "Max", 0.7)
        .link(noise, "Out", random, "X")
        .link(random, "Out", sunk, "A")
        .link(variation, "Out", sunk, "B")
        .value(top, "A", 1.0)
        .link(sunk, "Out", top, "B")
        .link(position, "Out", grain, "Position")
        .link(grain, "Out", grain_height, "A")
        .value(grain_height, "B", 0.1)
        .link(top, "Out", rough_top, "A")
        .link(grain_height, "Out", rough_top, "B")
        .link(grain_height, "Out", joint, "A")
        .value(joint, "B", 0.05)
        .link(joint, "Out", height, "A")
        .link(rough_top, "Out", height, "B")
        .link(brick, "Out", height, "Alpha")
        .link(brick_color, "Out", brick_rgb, "In")
        .value(tint, "A", 0.6)
        .value(tint, "B", 1.35)
        .link(random, "Out", tint, "Alpha")
        .link(brick_rgb, "Out", tinted, "A")
        .link(tint, "Out", tinted, "B")
        .link(mortar_color, "Out", base, "A")
        .link(tinted, "Out", base, "B")
        .link(brick, "Out", base, "Alpha")
        .value(roughness, "A", 0.95)
        .value(roughness, "B", 0.78)
        .link(brick, "Out", roughness, "Alpha")
        .link(base, "Out", output, "BaseColor")
        .link(roughness, "Out", output, "Roughness")
        .link(height, "Out", output, "Displacement");
    g.build()
}
