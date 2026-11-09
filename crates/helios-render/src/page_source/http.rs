//! Browser page source: `fetch` with HTTP `Range` requests.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use super::{PageRead, PageSource, ReadCompletion};

/// The body of a [`fetch`].
pub enum Fetched {
    /// The requested range (HTTP 206), or the whole resource when no range was asked for.
    Range(Vec<u8>),
    /// The server ignored the `Range` header and sent the whole resource (HTTP 200).
    Whole(Vec<u8>),
}

/// GETs `url`, optionally just bytes `range` of it.
pub async fn fetch(url: &str, range: Option<Range<u64>>) -> Result<Fetched, String> {
    let js = |e: wasm_bindgen::JsValue| format!("{url}: {e:?}");
    let init = web_sys::RequestInit::new();
    init.set_method("GET");
    if let Some(range) = &range {
        let headers = web_sys::Headers::new().map_err(js)?;
        headers.set("Range", &format!("bytes={}-{}", range.start, range.end.saturating_sub(1))).map_err(js)?;
        init.set_headers(&headers);
    }
    let request = web_sys::Request::new_with_str_and_init(url, &init).map_err(js)?;
    let window = web_sys::window().ok_or("no window")?;
    let response: web_sys::Response = JsFuture::from(window.fetch_with_request(&request)).await.map_err(js)?.dyn_into().map_err(js)?;
    let status = response.status();
    if status != 200 && status != 206 {
        return Err(format!("{url}: HTTP {status}"));
    }
    let body = JsFuture::from(response.array_buffer().map_err(js)?).await.map_err(js)?;
    let bytes = js_sys::Uint8Array::new(&body).to_vec();
    match range {
        Some(range) if status == 206 => {
            if bytes.len() as u64 != range.end - range.start {
                return Err(format!("{url}: asked for {} bytes, got {}", range.end - range.start, bytes.len()));
            }
            Ok(Fetched::Range(bytes))
        }
        Some(_) => Ok(Fetched::Whole(bytes)),
        None => Ok(Fetched::Range(bytes)),
    }
}

/// Streams page payloads from `.hvm` files over HTTP, one `Range` request per read.
///
/// Servers that ignore `Range` (e.g. `python -m http.server`) answer with the whole file: the
/// source then keeps that file and serves later reads of it from memory, so streaming still
/// works — it just degrades to downloading each file once.
pub struct HttpSource {
    urls: Vec<String>,
    completed: Rc<RefCell<Vec<ReadCompletion>>>,
    whole_files: Rc<RefCell<HashMap<u32, Rc<Vec<u8>>>>>,
}

impl HttpSource {
    /// File `i` of a [`PageRead`] is fetched from `urls[i]`.
    pub fn new(urls: Vec<String>) -> Self {
        Self { urls, completed: Rc::default(), whole_files: Rc::default() }
    }
}

fn slice(file: &[u8], range: &Range<u64>) -> Result<Vec<u8>, String> {
    file.get(range.start as usize..range.end as usize).map(<[u8]>::to_vec).ok_or_else(|| "range past the end of the file".to_string())
}

impl PageSource for HttpSource {
    fn start(&self, read: PageRead) {
        if let Some(file) = self.whole_files.borrow().get(&read.file) {
            self.completed.borrow_mut().push(ReadCompletion { ticket: read.ticket, result: slice(file, &read.range) });
            return;
        }
        let Some(url) = self.urls.get(read.file as usize).cloned() else {
            self.completed.borrow_mut().push(ReadCompletion { ticket: read.ticket, result: Err(format!("no file {}", read.file)) });
            return;
        };
        let (completed, whole_files) = (self.completed.clone(), self.whole_files.clone());
        wasm_bindgen_futures::spawn_local(async move {
            let result = match fetch(&url, Some(read.range.clone())).await {
                Ok(Fetched::Range(bytes)) => Ok(bytes),
                Ok(Fetched::Whole(file)) => {
                    let result = slice(&file, &read.range);
                    if whole_files.borrow_mut().insert(read.file, Rc::new(file)).is_none() {
                        log::warn!("{url}: server ignores HTTP Range requests; downloading whole files instead");
                    }
                    result
                }
                Err(e) => Err(e),
            };
            completed.borrow_mut().push(ReadCompletion { ticket: read.ticket, result });
        });
    }

    fn poll(&self, completed: &mut Vec<ReadCompletion>) {
        completed.append(&mut self.completed.borrow_mut());
    }

    /// Browsers open about six HTTP/1.1 connections per host.
    fn max_in_flight(&self) -> usize {
        6
    }

    /// A request costs a round trip; reading up to 256 KiB extra is cheaper.
    fn max_gap_bytes(&self) -> u64 {
        256 << 10
    }
}
