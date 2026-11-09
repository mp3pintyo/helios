//! URI helpers: embedded `data:` URIs and percent-decoding of relative file URIs.

use base64::Engine;

pub fn is_data_uri(uri: &str) -> bool {
    uri.starts_with("data:")
}

/// Decodes a base64 `data:` URI (`data:application/octet-stream;base64,...`).
pub fn decode_data_uri(uri: &str) -> Option<Vec<u8>> {
    let (header, payload) = uri.strip_prefix("data:")?.split_once(',')?;
    if !header.ends_with(";base64") {
        return None; // glTF only uses base64 payloads
    }
    base64::engine::general_purpose::STANDARD.decode(payload).ok()
}

/// Decodes `%XX` escapes (relative URIs in glTF are percent-encoded, e.g. spaces as `%20`).
/// Malformed escapes are kept verbatim. (Browsers resolve URIs themselves; only file paths need this.)
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub fn percent_decode(uri: &str) -> String {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_uris() {
        assert_eq!(decode_data_uri("data:application/octet-stream;base64,AAEC/w=="), Some(vec![0, 1, 2, 255]));
        assert_eq!(decode_data_uri("data:text/plain,hello"), None);
        assert!(is_data_uri("data:image/png;base64,"));
        assert!(!is_data_uri("textures/a.png"));
    }

    #[test]
    fn percent_decoding() {
        assert_eq!(percent_decode("My%20Texture%2B1.png"), "My Texture+1.png");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("a%zzb"), "a%zzb");
        assert_eq!(percent_decode("%C3%A9.jpg"), "é.jpg");
    }
}
