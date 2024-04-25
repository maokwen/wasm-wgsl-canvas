mod logger;
mod res;
mod utils;

use std::mem;

use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub async fn run() {
    logger::init_logger();

    let canvas_list = initialize_canvas();

    if canvas_list.is_empty() {
        log::info!("No canvas found");
        return;
    }

    let mut context_list: Vec<Context> = vec![];
    for canvas in canvas_list {
        let context = initialize_context(canvas).await;
        context_list.push(context);
    }

    let shader_desc = wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r##"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.position = vec2<f32>(x, y);
    return out;
}             

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.position, 0.5, 1.0);
}

"##
            .into(),
        ),
    };

    let mut idx = 0;

    for context in context_list {
        let (layout, perframe_layout, texture_layout, sampler_layout) =
            initialize_pipeline_layout(&context.device);
        let (perframe_bind_group, time_buffer, mouse_buffer) =
            initialize_bind_group_perframe(&context.device, &perframe_layout);
        let (texture_bind_group, channel0, view0, channel1, view1) =
            initialize_bind_group_texture(&context.device, &texture_layout);
        let sampler_bind_group = initialize_bind_group_sampler(&context.device, &sampler_layout);

        let shader = context.device.create_shader_module(shader_desc.clone());
        let render_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: context.config.format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        unclipped_depth: false,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                });

        // render

        let output = context.surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            static CLEAR_COLOR: wgpu::Color = wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            let time_uniform = TimeUniform {
                frame: 1,
                elapsed: 1.0,
                delta: 1.0,
            };

            let mouse_uniform = MouseUniform {
                pos: [0, 0],
                click: 0,
            };

            context
                .queue
                .write_buffer(&time_buffer, 0, bytemuck::cast_slice(&[time_uniform]));

            context
                .queue
                .write_buffer(&mouse_buffer, 0, bytemuck::cast_slice(&[mouse_uniform]));

            render_pass.set_pipeline(&render_pipeline);
            render_pass.set_bind_group(0, &perframe_bind_group, &[]);
            render_pass.set_bind_group(1, &texture_bind_group, &[]);
            render_pass.set_bind_group(2, &sampler_bind_group, &[]);
            render_pass.draw((0 + idx)..(3 + idx), 0..1);
            log::info!("Triangle: {:#?} {:#?} ", (0 + idx), (3 + idx));

            idx += 1;
        }

        // submit will accept anything that implements IntoIter
        context.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

pub fn initialize_canvas() -> Vec<web_sys::HtmlCanvasElement> {
    use wasm_bindgen::JsCast;

    let mut canvas_list: Vec<web_sys::HtmlCanvasElement> = vec![];
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let list = doc.get_elements_by_class_name("wgpu-canvas");
            for i in 0..list.length() {
                let canvas = list
                    .item(i)
                    .unwrap()
                    .dyn_into::<web_sys::HtmlCanvasElement>()
                    .expect("Failed to initialize canvas");
                canvas_list.push(canvas);
            }
            Some(canvas_list)
        })
        .unwrap()
}

struct Context<'canvas> {
    canvas: web_sys::HtmlCanvasElement,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'canvas>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
}

async fn initialize_context(canvas: web_sys::HtmlCanvasElement) -> Context<'static> {
    let (canvas, instance, surface, adapter, device, queue, config) = {
        let (width, height) = (canvas.width(), canvas.height());

        let instance = initialize_instance();

        let surface_target = wgpu::SurfaceTarget::Canvas(canvas.clone());
        let surface = instance
            .create_surface(surface_target)
            .expect("Could not create surface from canvas");

        log::info!("Surface info: {:#?}", surface);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..wgpu::RequestAdapterOptions::default()
            })
            .await
            .expect("Could not get adapter");

        log::info!("Adapter info: {:#?}", adapter.get_info());

        let (device, queue) = initialize_device(&adapter).await;

        log::info!("Device info: {:#?}", device);
        log::info!("Queue info: {:#?}", adapter);

        let config = surface
            .get_default_config(&adapter, width, height)
            .expect("Surface isn't supported by the adapter.");

        let surface_formats = surface.get_capabilities(&adapter).formats;
        log::info!("Support texture formats: {:#?}", surface_formats);

        log::info!("Surface configuration info: {:#?}", config);

        (canvas, instance, surface, adapter, device, queue, config)
    };

    surface.configure(&device, &config);

    Context {
        canvas,
        instance,
        surface,
        adapter,
        device,
        queue,
        config,
    }
}

pub fn initialize_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        flags: wgpu::InstanceFlags::debugging().with_env(),
        ..wgpu::InstanceDescriptor::default()
    })
}

pub async fn initialize_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let bundle = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(), //wgpu::Limits::downlevel_webgl2_defaults(),
            },
            None, // Trace path
        )
        .await;

    match bundle {
        Ok(b) => b,
        Err(e) => panic!("Failed to initialize device: {}", e),
    }
}

pub fn initialize_pipeline_layout(
    device: &wgpu::Device,
) -> (
    wgpu::PipelineLayout,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
    // group(0)
    // binding(0) var<uniform> _time: Time { frame, elapsed }
    // binding(1) var<uniform> _mouse: Mouse { pos, click }
    let perframe_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // group(1)
    // binding(0) var _channel0: texture_2d<f32>
    // binding(1) var _channel1: textute_2s<f32>
    let texture_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

    // group(2)
    // binding(0) var _nearest: sampler;
    // binding(1) var _nearest_repeat: sampler;
    // binding(2) var _bilinear: sampler;
    // binding(3) var _bilinear_repeat: sampler;
    let sampler_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sampler Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipline Layout"),
        bind_group_layouts: &[
            &perframe_bind_group_layout,
            &texture_bind_group_layout,
            &sampler_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    (
        layout,
        perframe_bind_group_layout,
        texture_bind_group_layout,
        sampler_bind_group_layout,
    )
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    frame: u32,
    elapsed: f32,
    delta: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MouseUniform {
    pos: [u32; 2],
    click: i32,
}

pub fn initialize_bind_group_perframe(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, wgpu::Buffer, wgpu::Buffer) {
    // binding(0) var<uniform> _time: Time { frame, elapsed }
    // binding(1) var<uniform> _mouse: Mouse { pos, click }

    let time_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("_time Buffer"),
        size: mem::size_of::<TimeUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mouse_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("_mouse Buffer"),
        size: mem::size_of::<MouseUniform>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Per-frame Information Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: time_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mouse_buffer.as_entire_binding(),
            },
        ],
    });

    (bind_group, time_buffer, mouse_buffer)
}

pub fn initialize_bind_group_texture(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (
    wgpu::BindGroup,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    // binding(0) var _channel0: texture_2d<f32>
    // binding(1) var _channel1: textute_2s<f32>

    let channel0_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("_channel0 Default Texture"),
        size: wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let channel1_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("_channel0 Default Texture"),
        size: wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let view0 = channel0_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let view1 = channel1_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Per-frame Information Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view0),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view1),
            },
        ],
    });

    (bind_group, channel0_texture, view0, channel1_texture, view1)
}

pub fn initialize_bind_group_sampler(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    // binding(0) var _nearest: sampler;
    // binding(1) var _nearest_repeat: sampler;
    // binding(2) var _bilinear: sampler;
    // binding(3) var _bilinear_repeat: sampler;
    let nearest_sample = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("_nearest Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bilinear_sample = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("_nearest Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let nearest_repeat_sample = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("_nearest Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bilinear_repeat_sample = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("_nearest Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Per-frame Information Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(&nearest_sample),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&bilinear_sample),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&nearest_repeat_sample),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&bilinear_repeat_sample),
            },
        ],
    });

    bind_group
}

enum ChannelNo {
    C0 = 0,
    C1 = 1,
}

pub async fn load_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    channel: ChannelNo,
    url: &str,
) -> anyhow::Result<wgpu::BindGroup> {
    let (texture_desc, texture_image, size) = res::load_texture_from_url(None, url).await?;

    let texture = device.create_texture(&texture_desc);

    queue.write_texture(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        &texture_image,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * size.width),
            rows_per_image: Some(size.height),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let channel_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout,
        label: None,
        entries: &[wgpu::BindGroupEntry {
            binding: channel as u32,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });

    anyhow::Ok(channel_bind_group)
}

pub fn on_canvas_resize() {}
