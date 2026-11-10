//! Example graphs (the gallery scene's and the editor's starting points).

use crate::graph::{GraphBuilder, MaterialGraph, Op, VectorSpace};

/// Preset ids and labels, for editors.
pub const PRESETS: [(&str, &str); 8] = [
    ("blank", "Blank (output only)"),
    ("lava", "Animated lava"),
    ("wet_stone", "Wet stone with puddles"),
    ("stylized_water", "Stylised water"),
    ("brick_pavement", "Brick pavement (displaced)"),
    ("mushrooms", "Mushrooms (vector displacement)"),
    ("rippling_cloth", "Rippling cloth (animated displacement)"),
    ("combed_tufts", "Combed tufts (tangent-space vector displacement)"),
];

/// The preset graph `id`.
pub fn preset(id: &str) -> Option<MaterialGraph> {
    match id {
        "blank" => Some(blank()),
        "lava" => Some(lava()),
        "wet_stone" => Some(wet_stone()),
        "stylized_water" => Some(stylized_water()),
        "brick_pavement" => Some(brick_pavement()),
        "mushrooms" => Some(mushrooms()),
        "rippling_cloth" => Some(rippling_cloth()),
        "combed_tufts" => Some(combed_tufts()),
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

/// Mushrooms growing out of a flat, mossy ground: a **vector** displacement (the
/// VectorDisplacement output) with real overhangs — caps wider than their stems, with an
/// underside. World xz is divided into staggered cells; around each cell's centre a disc of the
/// ground is wrapped over a mushroom's silhouette, from the middle outwards: the dome of the cap
/// (a quarter ellipse), its underside back in to the stem, down the stem, and the rest of the
/// disc stretched back out over the ground, so the surface stays continuous and the displacement
/// is zero at the disc's rim. A point at distance r from the centre moves to radius ρ(r) and
/// height y(r): sideways by its offset from the centre times (ρ / r − 1), and up by y. A
/// per-cell random value sizes each mushroom; "Stem Height" raises the caps; the colours follow
/// the same pieces of the silhouette (spotted cap, gills, stem, moss). Towards the edges of the
/// bed (mesh UVs 0..1 across it) the whole displacement fades out, so the border stays put and no
/// mushroom is cut in half.
/// Its material needs a displacement magnitude of [`MUSHROOM_MAGNITUDE`] (the vector is in units
/// of it; another magnitude scales the mushrooms) on a surface whose normal is +Y.
pub fn mushrooms() -> MaterialGraph {
    // Cell size, the disc's radius, cap and stem radius and the cap dome's height (metres).
    const CELL: f32 = 0.26;
    const DISC: f32 = 0.1;
    const CAP: f32 = 0.085;
    const STEM: f32 = 0.02;
    const DOME: f32 = 0.035;
    // The silhouette's pieces in r / DISC: dome up to A, underside to B, stem to C, then ground.
    const A: f32 = 0.5;
    const B: f32 = 0.7;
    const C: f32 = 0.9;
    let mut g = GraphBuilder::new("Mushrooms");
    // Staggered cells and the offset from the cell's centre in metres.
    let position = g.node(Op::WorldPosition, [-2300.0, 0.0]);
    let xz = g.node(Op::ComponentMask { channels: "xz".into() }, [-2130.0, 0.0]);
    let cells = g.node(Op::Multiply, [-1960.0, 0.0]);
    let row = g.node(Op::ComponentMask { channels: "y".into() }, [-1790.0, 80.0]);
    let row_index = g.node(Op::Floor, [-1620.0, 80.0]);
    let half_row = g.node(Op::Multiply, [-1450.0, 80.0]);
    let stagger = g.node(Op::Frac, [-1280.0, 80.0]);
    let column = g.node(Op::ComponentMask { channels: "x".into() }, [-1790.0, -40.0]);
    let shifted_column = g.node(Op::Add, [-1110.0, -40.0]);
    let shifted = g.node(Op::Append, [-940.0, 0.0]);
    let cell = g.node(Op::Floor, [-770.0, 200.0]);
    let local = g.node(Op::Frac, [-770.0, 0.0]);
    let centred = g.node(Op::Subtract, [-600.0, 0.0]);
    let offset = g.node(Op::Multiply, [-430.0, 0.0]);
    let radius = g.node(Op::Length, [-260.0, 60.0]);
    let safe_radius = g.node(Op::Max, [-90.0, 140.0]);
    let s = g.node(Op::Divide, [-90.0, 40.0]);
    // The silhouette's pieces as ramps 0..1.
    let dome_ramp = g.node(Op::Divide, [80.0, -160.0]);
    let dome = g.node(Op::Saturate, [250.0, -160.0]);
    let underside = g.node(Op::SmoothStep, [250.0, -60.0]);
    let stem = g.node(Op::SmoothStep, [250.0, 60.0]);
    let ground = g.node(Op::SmoothStep, [250.0, 180.0]);
    let angle = g.node(Op::Multiply, [420.0, -160.0]);
    let sine = g.node(Op::Sine, [590.0, -200.0]);
    let cosine = g.node(Op::Cosine, [590.0, -120.0]);
    // Per-cell size.
    let seed = g.node(Op::Multiply, [-600.0, 200.0]);
    let noise = g.node(Op::Noise { scale: 1.0, octaves: 1 }, [-430.0, 200.0]);
    let random = g.node(Op::SmoothStep, [-260.0, 200.0]);
    let size = g.node(Op::Lerp, [-90.0, 260.0]);
    // ρ: the cap's dome out to its rim, the underside back in to the stem, the ground back out.
    let cap_radius = g.node(Op::Multiply, [590.0, 260.0]);
    let cap_shape = g.node(Op::Subtract, [760.0, -200.0]);
    let cap_term = g.node(Op::Multiply, [930.0, -160.0]);
    let stem_term = g.node(Op::Multiply, [760.0, -40.0]);
    let beyond_stem = g.node(Op::Subtract, [590.0, 140.0]);
    let ground_term = g.node(Op::Multiply, [760.0, 140.0]);
    let rho_cap = g.node(Op::Add, [1100.0, -100.0]);
    let rho = g.node(Op::Add, [1270.0, -40.0]);
    let ratio = g.node(Op::Divide, [1440.0, -40.0]);
    let stretch = g.node(Op::Subtract, [1610.0, -40.0]);
    let sideways = g.node(Op::Multiply, [1780.0, 0.0]);
    // y: the dome over the stem.
    let stem_height = g.node(Op::ScalarParameter { name: "Stem Height".into(), default: 0.07, min: 0.02, max: 0.075 }, [590.0, 380.0]);
    let dome_height = g.node(Op::Multiply, [760.0, 300.0]);
    let up_stem = g.node(Op::OneMinus, [590.0, 480.0]);
    let stem_part = g.node(Op::Multiply, [760.0, 420.0]);
    let silhouette = g.node(Op::Add, [930.0, 360.0]);
    let height = g.node(Op::Multiply, [1100.0, 360.0]);
    let side_x = g.node(Op::ComponentMask { channels: "x".into() }, [1950.0, -60.0]);
    let side_z = g.node(Op::ComponentMask { channels: "y".into() }, [1950.0, 40.0]);
    let xy = g.node(Op::Append, [2120.0, 0.0]);
    let xyz = g.node(Op::Append, [2290.0, 40.0]);
    let vector = g.node(Op::Multiply, [2460.0, 40.0]);
    // Fade towards the bed's edges.
    let uv = g.node(Op::TexCoord { set: 0, tiling: [1.0, 1.0] }, [1610.0, 240.0]);
    let far_side = g.node(Op::OneMinus, [1780.0, 300.0]);
    let nearest = g.node(Op::Min, [1950.0, 240.0]);
    let inside = g.node(Op::SmoothStep, [2120.0, 240.0]);
    let inside_u = g.node(Op::ComponentMask { channels: "x".into() }, [2290.0, 200.0]);
    let inside_v = g.node(Op::ComponentMask { channels: "y".into() }, [2290.0, 300.0]);
    let fade = g.node(Op::Multiply, [2460.0, 240.0]);
    let faded = g.node(Op::Multiply, [2630.0, 100.0]);
    // Colours along the silhouette.
    let cap_color = g.node(Op::VectorParameter { name: "Cap Color".into(), default: [0.5, 0.05, 0.03, 1.0] }, [930.0, -620.0]);
    let cap_rgb = g.node(Op::ComponentMask { channels: "rgb".into() }, [1100.0, -620.0]);
    let spots = g.node(Op::Noise { scale: 85.0, octaves: 2 }, [930.0, -500.0]);
    let spot_mask = g.node(Op::SmoothStep, [1100.0, -500.0]);
    let spot_color = g.node(Op::Constant3 { value: [0.9, 0.86, 0.74] }, [1100.0, -400.0]);
    let spotted = g.node(Op::Lerp, [1270.0, -560.0]);
    let gills = g.node(Op::Constant3 { value: [0.62, 0.52, 0.4] }, [1270.0, -440.0]);
    let under = g.node(Op::SmoothStep, [1270.0, -340.0]);
    let cap_or_gills = g.node(Op::Lerp, [1440.0, -500.0]);
    let stem_color = g.node(Op::Constant3 { value: [0.82, 0.78, 0.66] }, [1440.0, -380.0]);
    let to_stem = g.node(Op::Lerp, [1610.0, -460.0]);
    let moss_noise = g.node(Op::Noise { scale: 7.0, octaves: 4 }, [1440.0, -260.0]);
    let moss = g.node(Op::Lerp, [1610.0, -300.0]);
    let moss_dark = g.node(Op::Constant3 { value: [0.05, 0.075, 0.025] }, [1440.0, -180.0]);
    let moss_light = g.node(Op::Constant3 { value: [0.16, 0.22, 0.06] }, [1440.0, -100.0]);
    let on_ground = g.node(Op::SmoothStep, [1610.0, -180.0]);
    let base = g.node(Op::Lerp, [1780.0, -400.0]);
    let roughness = g.node(Op::Lerp, [1780.0, -240.0]);
    let output = g.node(Op::Output, [2840.0, -200.0]);
    g.link(position, "Out", xz, "In")
        .link(xz, "Out", cells, "A")
        .value(cells, "B", 1.0 / CELL)
        .link(cells, "Out", row, "In")
        .link(row, "Out", row_index, "In")
        .link(row_index, "Out", half_row, "A")
        .value(half_row, "B", 0.5)
        .link(half_row, "Out", stagger, "In")
        .link(cells, "Out", column, "In")
        .link(column, "Out", shifted_column, "A")
        .link(stagger, "Out", shifted_column, "B")
        .link(shifted_column, "Out", shifted, "A")
        .link(row, "Out", shifted, "B")
        .link(shifted, "Out", cell, "In")
        .link(shifted, "Out", local, "In")
        .link(local, "Out", centred, "A")
        .value(centred, "B", 0.5)
        .link(centred, "Out", offset, "A")
        .value(offset, "B", CELL)
        .link(offset, "Out", radius, "In")
        .link(radius, "Out", safe_radius, "A")
        .value(safe_radius, "B", 1e-5)
        .link(radius, "Out", s, "A")
        .value(s, "B", DISC)
        .link(s, "Out", dome_ramp, "A")
        .value(dome_ramp, "B", A)
        .link(dome_ramp, "Out", dome, "In")
        .value(underside, "Min", A)
        .value(underside, "Max", B)
        .link(s, "Out", underside, "X")
        .value(stem, "Min", B)
        .value(stem, "Max", C)
        .link(s, "Out", stem, "X")
        .value(ground, "Min", C)
        .value(ground, "Max", 1.0)
        .link(s, "Out", ground, "X")
        .link(dome, "Out", angle, "A")
        .value(angle, "B", std::f32::consts::FRAC_PI_2)
        .link(angle, "Out", sine, "In")
        .link(angle, "Out", cosine, "In")
        .link(cell, "Out", seed, "A")
        .value(seed, "B", 0.613)
        .link(seed, "Out", noise, "Position")
        .value(random, "Min", 0.3)
        .value(random, "Max", 0.7)
        .link(noise, "Out", random, "X")
        .value(size, "A", 0.5)
        .value(size, "B", 1.0)
        .link(random, "Out", size, "Alpha")
        .link(size, "Out", cap_radius, "A")
        .value(cap_radius, "B", CAP)
        .link(sine, "Out", cap_shape, "A")
        .link(underside, "Out", cap_shape, "B")
        .link(cap_shape, "Out", cap_term, "A")
        .link(cap_radius, "Out", cap_term, "B")
        .link(underside, "Out", stem_term, "A")
        .value(stem_term, "B", STEM)
        .link(radius, "Out", beyond_stem, "A")
        .value(beyond_stem, "B", STEM)
        .link(ground, "Out", ground_term, "A")
        .link(beyond_stem, "Out", ground_term, "B")
        .link(cap_term, "Out", rho_cap, "A")
        .link(stem_term, "Out", rho_cap, "B")
        .link(rho_cap, "Out", rho, "A")
        .link(ground_term, "Out", rho, "B")
        .link(rho, "Out", ratio, "A")
        .link(safe_radius, "Out", ratio, "B")
        .link(ratio, "Out", stretch, "A")
        .value(stretch, "B", 1.0)
        .link(offset, "Out", sideways, "A")
        .link(stretch, "Out", sideways, "B")
        .link(cosine, "Out", dome_height, "A")
        .value(dome_height, "B", DOME)
        .link(stem, "Out", up_stem, "In")
        .link(up_stem, "Out", stem_part, "A")
        .link(stem_height, "Out", stem_part, "B")
        .link(dome_height, "Out", silhouette, "A")
        .link(stem_part, "Out", silhouette, "B")
        .link(silhouette, "Out", height, "A")
        .link(size, "Out", height, "B")
        .link(sideways, "Out", side_x, "In")
        .link(sideways, "Out", side_z, "In")
        .link(side_x, "Out", xy, "A")
        .link(height, "Out", xy, "B")
        .link(xy, "Out", xyz, "A")
        .link(side_z, "Out", xyz, "B")
        .link(xyz, "Out", vector, "A")
        .value(vector, "B", 1.0 / MUSHROOM_MAGNITUDE)
        .link(uv, "Out", far_side, "In")
        .link(uv, "Out", nearest, "A")
        .link(far_side, "Out", nearest, "B")
        .value(inside, "Min", 0.01)
        .value(inside, "Max", 0.1)
        .link(nearest, "Out", inside, "X")
        .link(inside, "Out", inside_u, "In")
        .link(inside, "Out", inside_v, "In")
        .link(inside_u, "Out", fade, "A")
        .link(inside_v, "Out", fade, "B")
        .link(vector, "Out", faded, "A")
        .link(fade, "Out", faded, "B")
        .link(cap_color, "Out", cap_rgb, "In")
        .link(position, "Out", spots, "Position")
        .value(spot_mask, "Min", 0.57)
        .value(spot_mask, "Max", 0.63)
        .link(spots, "Out", spot_mask, "X")
        .link(cap_rgb, "Out", spotted, "A")
        .link(spot_color, "Out", spotted, "B")
        .link(spot_mask, "Out", spotted, "Alpha")
        .value(under, "Min", A - 0.04)
        .value(under, "Max", A + 0.02)
        .link(s, "Out", under, "X")
        .link(spotted, "Out", cap_or_gills, "A")
        .link(gills, "Out", cap_or_gills, "B")
        .link(under, "Out", cap_or_gills, "Alpha")
        .link(cap_or_gills, "Out", to_stem, "A")
        .link(stem_color, "Out", to_stem, "B")
        .link(stem, "Out", to_stem, "Alpha")
        .link(position, "Out", moss_noise, "Position")
        .link(moss_dark, "Out", moss, "A")
        .link(moss_light, "Out", moss, "B")
        .link(moss_noise, "Out", moss, "Alpha")
        .link(to_stem, "Out", base, "A")
        .link(moss, "Out", base, "B")
        .value(on_ground, "Min", C - 0.02)
        .value(on_ground, "Max", C + 0.01)
        .link(s, "Out", on_ground, "X")
        .link(on_ground, "Out", base, "Alpha")
        .value(roughness, "A", 0.4)
        .value(roughness, "B", 0.9)
        .link(under, "Out", roughness, "Alpha")
        .link(base, "Out", output, "BaseColor")
        .link(roughness, "Out", output, "Roughness")
        .link(faded, "Out", output, "VectorDisplacement");
    g.build()
}

/// The displacement magnitude [`mushrooms`] is modelled for: its tallest mushroom's top moves
/// 10.5 cm.
pub const MUSHROOM_MAGNITUDE: f32 = 0.11;

/// A cloth stretched over a frame, rippling: an **animated** displacement (its height reads the
/// time). Two sine waves travel across world xz in different directions, a scrolling noise (a
/// Panner on the renderer's clock) wrinkles them, and the sheet is pinned along its edges (mesh
/// UVs 0..1 across it). The colour follows the height, crests lighter.
/// Its material needs a displacement magnitude (the gallery uses 16 cm, centre 0.5). Instances of
/// an animated displacement are dynamic shadow casters.
pub fn rippling_cloth() -> MaterialGraph {
    let mut g = GraphBuilder::new("Rippling cloth");
    let position = g.node(Op::WorldPosition, [-1200.0, 0.0]);
    let xz = g.node(Op::ComponentMask { channels: "xz".into() }, [-1030.0, 0.0]);
    let time = g.node(Op::Time, [-1030.0, 260.0]);
    // Two travelling waves.
    let long_direction = g.node(Op::Constant2 { value: [5.2, 2.9] }, [-860.0, -160.0]);
    let long_phase = g.node(Op::Dot, [-690.0, -120.0]);
    let long_clock = g.node(Op::Multiply, [-690.0, -20.0]);
    let long_travel = g.node(Op::Subtract, [-520.0, -80.0]);
    let long_wave = g.node(Op::Sine, [-350.0, -80.0]);
    let long_height = g.node(Op::Multiply, [-180.0, -80.0]);
    let short_direction = g.node(Op::Constant2 { value: [-4.5, 11.0] }, [-860.0, 100.0]);
    let short_phase = g.node(Op::Dot, [-690.0, 120.0]);
    let short_clock = g.node(Op::Multiply, [-690.0, 220.0]);
    let short_travel = g.node(Op::Add, [-520.0, 160.0]);
    let short_wave = g.node(Op::Sine, [-350.0, 160.0]);
    let short_height = g.node(Op::Multiply, [-180.0, 160.0]);
    // Scrolling wrinkles.
    let scroll = g.node(Op::Panner { speed: [0.22, 0.09] }, [-860.0, 380.0]);
    let wrinkles = g.node(Op::Noise { scale: 4.0, octaves: 3 }, [-690.0, 380.0]);
    let wrinkle_centre = g.node(Op::Subtract, [-520.0, 380.0]);
    let wrinkle_height = g.node(Op::Multiply, [-350.0, 380.0]);
    let waves = g.node(Op::Add, [-10.0, 40.0]);
    let moving = g.node(Op::Add, [160.0, 120.0]);
    // Pinned edges.
    let uv = g.node(Op::TexCoord { set: 0, tiling: [1.0, 1.0] }, [-520.0, 560.0]);
    let far_side = g.node(Op::OneMinus, [-350.0, 620.0]);
    let nearest = g.node(Op::Min, [-180.0, 560.0]);
    let free = g.node(Op::SmoothStep, [-10.0, 560.0]);
    let free_u = g.node(Op::ComponentMask { channels: "x".into() }, [160.0, 520.0]);
    let free_v = g.node(Op::ComponentMask { channels: "y".into() }, [160.0, 620.0]);
    let pin = g.node(Op::Multiply, [330.0, 560.0]);
    let pinned = g.node(Op::Multiply, [500.0, 200.0]);
    let height = g.node(Op::Add, [670.0, 200.0]);
    // Colour by height.
    let color = g.node(Op::VectorParameter { name: "Cloth Color".into(), default: [0.08, 0.2, 0.55, 1.0] }, [330.0, -260.0]);
    let rgb = g.node(Op::ComponentMask { channels: "rgb".into() }, [500.0, -260.0]);
    let shade = g.node(Op::Lerp, [670.0, -120.0]);
    let base = g.node(Op::Multiply, [840.0, -200.0]);
    let output = g.node(Op::Output, [1040.0, 0.0]);
    g.link(position, "Out", xz, "In")
        .link(xz, "Out", long_phase, "A")
        .link(long_direction, "Out", long_phase, "B")
        .link(time, "Out", long_clock, "A")
        .value(long_clock, "B", 1.9)
        .link(long_phase, "Out", long_travel, "A")
        .link(long_clock, "Out", long_travel, "B")
        .link(long_travel, "Out", long_wave, "In")
        .link(long_wave, "Out", long_height, "A")
        .value(long_height, "B", 0.27)
        .link(xz, "Out", short_phase, "A")
        .link(short_direction, "Out", short_phase, "B")
        .link(time, "Out", short_clock, "A")
        .value(short_clock, "B", 3.1)
        .link(short_phase, "Out", short_travel, "A")
        .link(short_clock, "Out", short_travel, "B")
        .link(short_travel, "Out", short_wave, "In")
        .link(short_wave, "Out", short_height, "A")
        .value(short_height, "B", 0.1)
        .link(xz, "Out", scroll, "UV")
        .link(scroll, "Out", wrinkles, "Position")
        .link(wrinkles, "Out", wrinkle_centre, "A")
        .value(wrinkle_centre, "B", 0.5)
        .link(wrinkle_centre, "Out", wrinkle_height, "A")
        .value(wrinkle_height, "B", 0.24)
        .link(long_height, "Out", waves, "A")
        .link(short_height, "Out", waves, "B")
        .link(waves, "Out", moving, "A")
        .link(wrinkle_height, "Out", moving, "B")
        .link(uv, "Out", far_side, "In")
        .link(uv, "Out", nearest, "A")
        .link(far_side, "Out", nearest, "B")
        .value(free, "Min", 0.0)
        .value(free, "Max", 0.12)
        .link(nearest, "Out", free, "X")
        .link(free, "Out", free_u, "In")
        .link(free, "Out", free_v, "In")
        .link(free_u, "Out", pin, "A")
        .link(free_v, "Out", pin, "B")
        .link(moving, "Out", pinned, "A")
        .link(pin, "Out", pinned, "B")
        .link(pinned, "Out", height, "A")
        .value(height, "B", 0.5)
        .link(color, "Out", rgb, "In")
        .value(shade, "A", 0.7)
        .value(shade, "B", 1.3)
        .link(height, "Out", shade, "Alpha")
        .link(rgb, "Out", base, "A")
        .link(shade, "Out", base, "B")
        .link(base, "Out", output, "BaseColor")
        .link(height, "Out", output, "Displacement");
    g.build()
}

/// Combed tufts: a field of pointed tufts in a grid of object-space cells, each bending over
/// towards the surface's tangent (object X on a flat bed) by a "Lean" parameter. The vector is
/// built in tangent space — up the normal, along the tangent — and turned into world space by a
/// TransformVector node, so the tufts follow their instance: on a turning platform they turn with
/// it, where a world-space vector from the same constants would keep leaning the same way. The
/// tips are lighter than the roots, and the tufts shrink towards the mesh's edges (by its UVs).
/// Its material's magnitude is the tufts' height.
pub fn combed_tufts() -> MaterialGraph {
    let mut g = GraphBuilder::new("Combed tufts");
    let position = g.node(Op::ObjectPosition, [-1200.0, 0.0]);
    let xz = g.node(Op::ComponentMask { channels: "xz".into() }, [-1030.0, 0.0]);
    let density = g.node(Op::ScalarParameter { name: "Tuft Density".into(), default: 3.0, min: 1.0, max: 8.0 }, [-1030.0, 120.0]);
    let scaled = g.node(Op::Multiply, [-860.0, 40.0]);
    let cell = g.node(Op::Frac, [-690.0, 40.0]);
    let centred = g.node(Op::Subtract, [-520.0, 40.0]);
    let distance = g.node(Op::Length, [-350.0, 40.0]);
    let falloff = g.node(Op::SmoothStep, [-180.0, 40.0]);
    let profile = g.node(Op::OneMinus, [-10.0, 40.0]);
    let squared = g.node(Op::Multiply, [160.0, 40.0]);
    // Shorter towards the mesh's edges (by its UVs), so no tuft is cut in half.
    let uv = g.node(Op::TexCoord { set: 0, tiling: [1.0, 1.0] }, [-520.0, 300.0]);
    let far_side = g.node(Op::OneMinus, [-350.0, 360.0]);
    let nearest = g.node(Op::Min, [-180.0, 300.0]);
    let border = g.node(Op::SmoothStep, [-10.0, 300.0]);
    let border_u = g.node(Op::ComponentMask { channels: "x".into() }, [160.0, 260.0]);
    let border_v = g.node(Op::ComponentMask { channels: "y".into() }, [160.0, 340.0]);
    let inside = g.node(Op::Multiply, [330.0, 300.0]);
    let height = g.node(Op::Multiply, [330.0, 40.0]);
    // Up the normal; along the tangent by lean · height² (the tips bend furthest).
    let lean = g.node(Op::ScalarParameter { name: "Lean".into(), default: 0.8, min: 0.0, max: 1.4 }, [160.0, 200.0]);
    let bend = g.node(Op::Multiply, [330.0, 160.0]);
    let along = g.node(Op::Multiply, [500.0, 160.0]);
    let up = g.node(Op::Multiply, [500.0, 40.0]);
    let tangent_plane = g.node(Op::Append, [670.0, 120.0]);
    let tangent_vector = g.node(Op::Append, [840.0, 80.0]);
    let turn = g.node(Op::TransformVector { source: VectorSpace::Tangent, destination: VectorSpace::World }, [1010.0, 80.0]);
    // Lighter tips.
    let roots = g.node(Op::VectorParameter { name: "Root Color".into(), default: [0.05, 0.09, 0.03, 1.0] }, [840.0, -220.0]);
    let tips = g.node(Op::VectorParameter { name: "Tip Color".into(), default: [0.55, 0.5, 0.18, 1.0] }, [840.0, -120.0]);
    let colors = g.node(Op::Lerp, [1010.0, -160.0]);
    let rgb = g.node(Op::ComponentMask { channels: "rgb".into() }, [1100.0, -160.0]);
    let output = g.node(Op::Output, [1250.0, 0.0]);
    g.link(position, "Out", xz, "In")
        .link(xz, "Out", scaled, "A")
        .link(density, "Out", scaled, "B")
        .link(scaled, "Out", cell, "In")
        .link(cell, "Out", centred, "A")
        .value(centred, "B", 0.5)
        .link(centred, "Out", distance, "In")
        .value(falloff, "Min", 0.0)
        .value(falloff, "Max", 0.42)
        .link(distance, "Out", falloff, "X")
        .link(falloff, "Out", profile, "In")
        .link(profile, "Out", squared, "A")
        .link(profile, "Out", squared, "B")
        .link(uv, "Out", far_side, "In")
        .link(uv, "Out", nearest, "A")
        .link(far_side, "Out", nearest, "B")
        .value(border, "Min", 0.02)
        .value(border, "Max", 0.1)
        .link(nearest, "Out", border, "X")
        .link(border, "Out", border_u, "In")
        .link(border, "Out", border_v, "In")
        .link(border_u, "Out", inside, "A")
        .link(border_v, "Out", inside, "B")
        .link(squared, "Out", height, "A")
        .link(inside, "Out", height, "B")
        .link(height, "Out", bend, "A")
        .link(height, "Out", bend, "B")
        .link(bend, "Out", along, "A")
        .link(lean, "Out", along, "B")
        .link(height, "Out", up, "A")
        .value(up, "B", 0.7)
        .link(along, "Out", tangent_plane, "A")
        .value(tangent_plane, "B", 0.0)
        .link(tangent_plane, "Out", tangent_vector, "A")
        .link(up, "Out", tangent_vector, "B")
        .link(tangent_vector, "Out", turn, "In")
        .link(turn, "Out", output, "VectorDisplacement")
        .link(roots, "Out", colors, "A")
        .link(tips, "Out", colors, "B")
        .link(height, "Out", colors, "Alpha")
        .link(colors, "Out", rgb, "In")
        .link(rgb, "Out", output, "BaseColor");
    g.build()
}
