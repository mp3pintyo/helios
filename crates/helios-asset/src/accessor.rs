//! Vertex attribute accessors → `f32` components.
//!
//! The `gltf` crate's typed readers assume the core formats (float positions and normals,
//! unsigned normalized texture coordinates). `KHR_mesh_quantization` also allows signed and
//! unsigned 8- and 16-bit integers, normalized or not, for positions, normals and texture
//! coordinates (the node transform or `KHR_texture_transform` undoes the quantisation), so
//! attributes are decoded here, for every component type, interleaved or tightly packed, with
//! sparse substitution.

use std::borrow::Cow;

use gltf::accessor::DataType;

/// The accessor's elements as `f32` components, `accessor.dimensions().multiplicity()` per
/// element; `None` if it reaches outside its buffer.
pub fn read_floats(accessor: &gltf::Accessor, buffers: &[Cow<[u8]>]) -> Option<Vec<f32>> {
    let layout = Layout {
        data_type: accessor.data_type(),
        components: accessor.dimensions().multiplicity(),
        normalized: accessor.normalized(),
        count: accessor.count(),
    };
    let mut values = match accessor.view() {
        Some(view) => {
            let buffer = buffers.get(view.buffer().index())?;
            let data = buffer.get(view.offset()..view.offset() + view.length())?;
            layout.decode(data.get(accessor.offset()..)?, view.stride().unwrap_or(layout.element_bytes()))?
        }
        // No buffer view: all zeros, unless sparse values replace them.
        None => vec![0.0; layout.count * layout.components],
    };
    if let Some(sparse) = accessor.sparse() {
        let slice = |view: gltf::buffer::View, offset: usize| {
            buffers.get(view.buffer().index())?.get(view.offset()..view.offset() + view.length())?.get(offset..)
        };
        let indices = sparse.indices();
        let index_type = match indices.index_type() {
            gltf::accessor::sparse::IndexType::U8 => DataType::U8,
            gltf::accessor::sparse::IndexType::U16 => DataType::U16,
            gltf::accessor::sparse::IndexType::U32 => DataType::U32,
        };
        let index_layout = Layout { data_type: index_type, components: 1, normalized: false, count: sparse.count() };
        let targets = index_layout.decode(slice(indices.view(), indices.offset())?, index_type.size())?;
        let replacement_layout = Layout { count: sparse.count(), ..layout };
        let replacements = replacement_layout.decode(slice(sparse.values().view(), sparse.values().offset())?, layout.element_bytes())?;
        apply_sparse(&mut values, layout.components, &targets, &replacements)?;
    }
    Some(values)
}

/// How an accessor's elements are stored.
#[derive(Clone, Copy, Debug)]
struct Layout {
    data_type: DataType,
    components: usize,
    normalized: bool,
    count: usize,
}

impl Layout {
    fn element_bytes(&self) -> usize {
        self.data_type.size() * self.components
    }

    /// Decodes `count` elements starting at `data`, `stride` bytes apart; `None` if they do not
    /// fit.
    fn decode(&self, data: &[u8], stride: usize) -> Option<Vec<f32>> {
        if self.count == 0 {
            return Some(Vec::new());
        }
        let size = self.data_type.size();
        if stride < self.element_bytes() || data.len() < (self.count - 1) * stride + self.element_bytes() {
            return None;
        }
        let mut out = Vec::with_capacity(self.count * self.components);
        for element in 0..self.count {
            for c in 0..self.components {
                let at = element * stride + c * size;
                out.push(component(&data[at..at + size], self.data_type, self.normalized));
            }
        }
        Some(out)
    }
}

/// One component, converted as the glTF specification defines normalized integers
/// (`max(c / 127, -1)` for signed bytes, `c / 255` for unsigned ones, likewise for shorts).
fn component(bytes: &[u8], data_type: DataType, normalized: bool) -> f32 {
    let (value, max) = match data_type {
        DataType::I8 => (bytes[0] as i8 as f32, 127.0),
        DataType::U8 => (bytes[0] as f32, 255.0),
        DataType::I16 => (i16::from_le_bytes([bytes[0], bytes[1]]) as f32, 32767.0),
        DataType::U16 => (u16::from_le_bytes([bytes[0], bytes[1]]) as f32, 65535.0),
        DataType::U32 => return u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32,
        DataType::F32 => return f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    };
    if normalized {
        (value / max).max(-1.0)
    } else {
        value
    }
}

/// Replaces the elements listed in `targets` (element indices, as floats) by `replacements`.
fn apply_sparse(values: &mut [f32], components: usize, targets: &[f32], replacements: &[f32]) -> Option<()> {
    for (k, &target) in targets.iter().enumerate() {
        let at = target as usize * components;
        values.get_mut(at..at + components)?.copy_from_slice(&replacements[k * components..(k + 1) * components]);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(data_type: DataType, components: usize, normalized: bool, count: usize) -> Layout {
        Layout { data_type, components, normalized, count }
    }

    #[test]
    fn normalized_integers_follow_the_specification() {
        assert_eq!(component(&[127], DataType::I8, true), 1.0);
        assert_eq!(component(&[0x80], DataType::I8, true), -1.0, "-128 clamps to -1");
        assert_eq!(component(&[0x81], DataType::I8, true), -1.0);
        assert_eq!(component(&[255], DataType::U8, true), 1.0);
        assert_eq!(component(&32767i16.to_le_bytes(), DataType::I16, true), 1.0);
        assert_eq!(component(&(-16384i16).to_le_bytes(), DataType::I16, true), -16384.0 / 32767.0);
        assert_eq!(component(&65535u16.to_le_bytes(), DataType::U16, true), 1.0);
        // KHR_mesh_quantization: unnormalized integers keep their value (a transform scales them).
        assert_eq!(component(&40000u16.to_le_bytes(), DataType::U16, false), 40000.0);
        assert_eq!(component(&[0xfe], DataType::I8, false), -2.0);
        assert_eq!(component(&1.5f32.to_le_bytes(), DataType::F32, false), 1.5);
    }

    #[test]
    fn interleaved_elements_are_read_at_their_stride() {
        // Two i16x3 positions (6 bytes) padded to an 8-byte stride, interleaved with other data.
        let mut data = Vec::new();
        for p in [[1i16, -2, 3], [400, 500, -600]] {
            p.iter().for_each(|c| data.extend_from_slice(&c.to_le_bytes()));
            data.extend_from_slice(&[0xaa, 0xbb]);
        }
        let values = layout(DataType::I16, 3, false, 2).decode(&data, 8).unwrap();
        assert_eq!(values, [1.0, -2.0, 3.0, 400.0, 500.0, -600.0]);
        // The last element may end before the stride does, but not past the data.
        assert!(layout(DataType::I16, 3, false, 2).decode(&data[..14], 8).is_some());
        assert!(layout(DataType::I16, 3, false, 2).decode(&data[..13], 8).is_none());
        assert!(layout(DataType::I16, 3, false, 2).decode(&data, 4).is_none(), "stride below the element size");
    }

    #[test]
    fn sparse_values_replace_their_elements() {
        let mut values = vec![0.0; 8];
        apply_sparse(&mut values, 2, &[3.0, 1.0], &[7.0, 8.0, 5.0, 6.0]).unwrap();
        assert_eq!(values, [0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0, 8.0]);
        assert!(apply_sparse(&mut values, 2, &[4.0], &[1.0, 1.0]).is_none(), "out of range");
    }
}
