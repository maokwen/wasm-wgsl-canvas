mod logger;
mod utils;

use wasm_bindgen::{prelude::wasm_bindgen, JsCast};

#[wasm_bindgen]
pub async fn run() {
    logger::init_logger();

    let mut canvas_list: Vec<web_sys::HtmlCanvasElement> = vec![];
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let list = doc.get_elements_by_class_name("wgpu-canvas");
            for i in 0..list.length() {
                let parent = list.item(i).unwrap();
                let canvas = initialize_canvas(&parent);
                canvas_list.push(canvas);
            }
            Some(())
        })
        .expect("Failed to get canvas list");

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
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}             

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.3, 0.2, 0.1, 1.0);
}

"##
            .into(),
        ),
    };

    for context in context_list {
        let layout = initialize_pipeline_layout(&context.device);
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

            //render_pass.set_pipeline(&render_pipeline);
            //render_pass.draw(0..3, 0..1);
        }

        // submit will accept anything that implements IntoIter
        context.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

pub fn initialize_canvas(element: &web_sys::Element) -> web_sys::HtmlCanvasElement {
    use wasm_bindgen::JsCast;

    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            doc.create_element("canvas")
                .and_then(|e| element.append_child(&e))
                .ok()
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        })
        .expect("Failed to initialize canvas")
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
            .expect("could not create surface from canvas");

        log::info!("Surface info: {:#?}", surface);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..wgpu::RequestAdapterOptions::default()
            })
            .await
            .unwrap();

        log::info!("Adapter info: {:#?}", adapter.get_info());

        let (device, queue) = initialize_device(&adapter).await;

        log::info!("Device info: {:#?}", device);
        log::info!("Queue info: {:#?}", adapter);

        let config = surface
            .get_default_config(&adapter, width, height)
            .expect("Surface isn't supported by the adapter.");

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
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            },
            None, // Trace path
        )
        .await;

    match bundle {
        Ok(b) => b,
        Err(_e) => panic!("Failed to initialize device: {_e}"),
    }
}

pub fn initialize_pipeline_layout(device: &wgpu::Device) -> wgpu::PipelineLayout {
    // binding(0) iTime
    // binding(1) iTimeDelta
    // binding(2) iFrame
    let perframe_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Per-frame Information Bind Group Layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

    // binding(0) iResolution
    let info_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Static Information Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    // binging(0) iMouse
    let input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Input Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    // binding(0) iDate
    let date_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Date Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    // binding(0) iChannel
    // binding(1) iSampler
    let channel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Channels Bind Group Layout"),
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
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipline Layout"),
        bind_group_layouts: &[
            // &perframe_layout,
            // &info_layout,
            //&input_layout,
            //&date_layout,
            //&channel_layout,
        ],
        push_constant_ranges: &[],
    });

    layout
}

pub fn initialize_perframe_buffers(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let time_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iTime Buffer"),
        size: std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let timedelta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iTimeDelta Buffer"),
        size: std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iFrame Buffer"),
        size: std::mem::size_of::<f32>() as u64,
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
                resource: timedelta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: frame_buffer.as_entire_binding(),
            },
        ],
    });

    (bind_group, time_buffer, timedelta_buffer, frame_buffer)
}

pub fn initialize_info_buffers(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, wgpu::Buffer) {
    let resolution_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iResolution Buffer"),
        size: std::mem::size_of::<[i32; 3]>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Static Information Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: resolution_buffer.as_entire_binding(),
        }],
    });

    (bind_group, resolution_buffer)
}

pub fn initialize_input_buffers(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, wgpu::Buffer) {
    let mouse_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iMouse Buffer"),
        size: std::mem::size_of::<[f32; 4]>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Input Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: mouse_buffer.as_entire_binding(),
        }],
    });

    (bind_group, mouse_buffer)
}

pub fn initialize_date_buffers(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, wgpu::Buffer) {
    let date_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("iDate Buffer"),
        size: std::mem::size_of::<[f32; 4]>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Date Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: date_buffer.as_entire_binding(),
        }],
    });

    (bind_group, date_buffer)
}
