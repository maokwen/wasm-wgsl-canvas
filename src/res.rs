use image::GenericImageView;

pub async fn load_texture_from_url<'a>(
    label: Option<&'a str>,
    url: &str,
) -> anyhow::Result<(
    wgpu::TextureDescriptor<'a>,
    image::RgbaImage,
    wgpu::Extent3d,
)> {
    let url = reqwest::Url::parse(url)?;
    let bytes = reqwest::get(url).await?.bytes().await?.to_vec();
    let img = image::load_from_memory(&bytes)?;
    let dimensions = img.dimensions();
    let rgba = img.to_rgba8();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let desc = wgpu::TextureDescriptor {
        label,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };

    anyhow::Ok((desc, rgba, size))
}
