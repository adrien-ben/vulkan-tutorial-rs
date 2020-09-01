use std::io::Cursor;
use std::path::Path;

#[cfg(not(target_os = "android"))]
pub fn load<P: AsRef<Path>>(path: P) -> Cursor<Vec<u8>> {
    use std::fs::File;
    use std::io::Read;

    let mut buf = Vec::new();
    let fullpath = &Path::new("assets").join(&path);
    let mut file = File::open(&fullpath).unwrap();
    file.read_to_end(&mut buf).unwrap();
    Cursor::new(buf)
}

#[cfg(target_os = "android")]
pub fn load<P: AsRef<Path>>(path: P) -> Cursor<Vec<u8>> {
    use std::io::Read;

    let asset_manager = ndk_glue::native_activity().asset_manager();

    let path = path.as_ref().to_str().unwrap();
    let mut asset = asset_manager
        .open(&std::ffi::CString::new(path).unwrap())
        .unwrap();

    let mut buf = Vec::new();
    asset.read_to_end(&mut buf).unwrap();
    Cursor::new(buf)
}
