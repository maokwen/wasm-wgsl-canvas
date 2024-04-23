pub fn init_logger() {
    // As we don't have an environment to pull logging level from, we use the query string.
    let query_string = web_sys::window().unwrap().location().search().unwrap();
    let query_level: Option<log::LevelFilter> =
        parse_url_query_string(&query_string, "RUST_LOG").and_then(|x| x.parse().ok());

    // We keep wgpu at Error level, as it's very noisy.
    let base_level = query_level.unwrap_or(log::LevelFilter::Info);
    let wgpu_level = query_level.unwrap_or(log::LevelFilter::Error);

    // On web, we use fern, as console_log doesn't have filtering on a per-module level.
    fern::Dispatch::new()
        .level(base_level)
        .level_for("wgpu_core", wgpu_level)
        .level_for("wgpu_hal", wgpu_level)
        .level_for("naga", wgpu_level)
        .chain(fern::Output::call(console_log::log))
        .apply()
        .unwrap();
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

/// Parse the query string as returned by `web_sys::window()?.location().search()?` and get a
/// specific key out of it.
pub fn parse_url_query_string<'a>(query: &'a str, search_key: &str) -> Option<&'a str> {
    let query_string = query.strip_prefix('?')?;

    for pair in query_string.split('&') {
        let mut pair = pair.split('=');
        let key = pair.next()?;
        let value = pair.next()?;

        if key == search_key {
            return Some(value);
        }
    }

    None
}
