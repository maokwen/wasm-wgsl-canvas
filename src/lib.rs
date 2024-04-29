mod logger;
mod res;
mod utils;

use std::cell::RefCell;
use std::rc::Rc;

use std::mem;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsCast;

const TARGET_FPS: i32 = 60;

const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.1,
    g: 0.2,
    b: 0.3,
    a: 1.0,
};

#[wasm_bindgen]
pub async fn run() -> Result<(), wasm_bindgen::JsValue> {
    logger::init_logger();
    utils::set_panic_hook();

    let canvas_list = initialize_canvas();

    if canvas_list.is_empty() {
        log::info!("No canvas found");
        return Ok(());
    }

    let mut context_list: Vec<Context> = vec![];

    for canvas in canvas_list {
        let context = Context::new(canvas).await;
        context_list.push(context);
    }

    let time_begin = web_time::Instant::now();
    let mut frame: u32 = 0;
    let mut time_last = web_time::Instant::now();

    let mut draw_frame = move || {
        for context in &context_list {
            context.input();
            context.update();
        }

        // rendering
        for context in &mut context_list {
            match context.render() {
                Ok(_) => {}
                // Err(wgpu::SurfaceError::Outdated) => context.resize(),
                Err(e) => log::error!("{:?}", e),
            };
        }

        let time_delta = time_last.elapsed().as_secs_f32();
        time_last = web_time::Instant::now();

        let time_elapsed = time_begin.elapsed().as_secs_f32();
        context_list.iter_mut().for_each(|context| {
            context.time_uniform = TimeUniform {
                frame,
                elapsed: time_elapsed,
                delta: time_delta,
            };
        });

        frame += 1;
    };

    // callbacks for request_animation_frame
    let animate_f = Rc::new(RefCell::new(None));
    let animate_g = animate_f.clone();

    // callbacks for set_timeout
    let timeout_f = Rc::new(RefCell::new(None));
    let timeout_g = timeout_f.clone();

    *timeout_g.borrow_mut() = Some(Closure::new(move || {
        request_animation_frame(window(), animate_f.borrow().as_ref().unwrap());
    }));

    *animate_g.borrow_mut() = Some(Closure::new(move || {
        draw_frame();
        let extra_timeout = 1000 / TARGET_FPS;

        set_timeout(
            window(),
            timeout_f.borrow().as_ref().unwrap(),
            extra_timeout,
        );
    }));

    request_animation_frame(window(), timeout_g.borrow().as_ref().unwrap());
    Ok(())
}

fn window() -> web_sys::Window {
    web_sys::window().expect("no global `window` exists")
}

fn set_timeout(window: web_sys::Window, callback: &Closure<dyn FnMut()>, timeout: i32) {
    window.set_timeout_with_callback_and_timeout_and_arguments_0(
        callback.as_ref().unchecked_ref(),
        timeout,
    );
}

fn request_animation_frame(window: web_sys::Window, callback: &Closure<dyn FnMut()>) {
    window
        .request_animation_frame(callback.as_ref().unchecked_ref())
        .expect("should register `requestAnimationFrame` OK");
}
struct Context<'canvas> {
    canvas: web_sys::HtmlCanvasElement,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'canvas>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    pipeline_layout: wgpu::PipelineLayout,
    perframe_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    sampler_bind_group: wgpu::BindGroup,
    time_buffer: wgpu::Buffer,
    mouse_buffer: wgpu::Buffer,
    texture_bindgroup_layout: wgpu::BindGroupLayout,
    channel0_texture: wgpu::Texture,
    channel1_texture: wgpu::Texture,
    channel0_view: wgpu::TextureView,
    channel1_view: wgpu::TextureView,
    time_uniform: TimeUniform,
    mouse_uniform: Rc<RefCell<MouseUniform>>,
}

impl Context<'_> {
    async fn new(canvas: web_sys::HtmlCanvasElement) -> Self {
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

        // setup

        let shader_desc = wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                r##"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
};

struct Time { frame: u32, elapsed: f32, delta: f32 }
struct Mouse { pos: vec2<u32>, click: u32 }

@group(0) @binding(0)   var<uniform> _time: Time;
@group(0) @binding(1)   var<uniform> _mouse: Mouse;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;

    let elapsed = _time.elapsed;
    let rx = (cos(elapsed) * x) - (sin(elapsed) * y) + f32(_mouse.pos.x) / 1000.0;
    let ry = (cos(elapsed) * y) + (sin(elapsed) * x) + f32(_mouse.pos.y) / 1000.0;
    
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.position = vec2<f32>(rx, ry);
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

        let (pipeline_layout, perframe_layout, texture_layout, sampler_layout) =
            initialize_pipeline_layout(&device);
        let (perframe_bind_group, time_buffer, mouse_buffer) =
            initialize_bind_group_perframe(&device, &perframe_layout);
        let (texture_bind_group, channel0_texture, channel0_view, channel1_texture, channel1_view) =
            initialize_bind_group_texture(&device, &texture_layout);
        let sampler_bind_group = initialize_bind_group_sampler(&device, &sampler_layout);

        let shader = device.create_shader_module(shader_desc.clone());
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions {
                    ..Default::default()
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions {
                    ..Default::default()
                },
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

        let time_uniform = TimeUniform {
            frame: 1,
            elapsed: 0.0,
            delta: 0.0,
        };

        let mouse_uniform = Rc::new(RefCell::new(MouseUniform {
            pos: [0, 0],
            click: 0,
            _padding: 0,
        }));

        initialize_mouse_handler(&window(), mouse_uniform.clone());

        Context {
            canvas,
            instance,
            surface,
            adapter,
            device,
            queue,
            config,
            render_pipeline,
            pipeline_layout,
            perframe_bind_group,
            texture_bind_group,
            sampler_bind_group,
            time_buffer,
            mouse_buffer,
            texture_bindgroup_layout: texture_layout,
            channel0_texture,
            channel1_texture,
            channel0_view,
            channel1_view,
            time_uniform,
            mouse_uniform,
        }
    }

    fn config_shader(&mut self, shader: &str) {
        let shader_desc = wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        };

        let shader = self.device.create_shader_module(shader_desc.clone());

        let render_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        ..Default::default()
                    },
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        ..Default::default()
                    },
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

        self.render_pipeline = render_pipeline;

        // todo: reload
    }

    async fn config_texture(&mut self, channel: ChannelNo, url: &str) -> anyhow::Result<()> {
        let bind_group = load_image(
            &self.device,
            &self.queue,
            &self.texture_bindgroup_layout,
            channel,
            url,
        )
        .await?;

        self.texture_bind_group = bind_group;

        // todo: reload

        Ok(())
    }

    fn resize(&mut self) {
        log::info!("Resizing canvas");
        let width = self.canvas.client_width() as u32;
        let height = self.canvas.client_height() as u32;
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    fn input(&self) {}

    fn update(&self) {}

    fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

            self.queue.write_buffer(
                &self.time_buffer,
                0,
                bytemuck::cast_slice(&[self.time_uniform]),
            );

            let mut mouse: MouseUniform = MouseUniform {
                pos: [0, 0],
                click: 0,
                _padding: 0,
            };
            self.mouse_uniform.as_ref().borrow().clone_into(&mut mouse);
            self.queue
                .write_buffer(&self.mouse_buffer, 0, bytemuck::cast_slice(&[mouse]));

            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.perframe_bind_group, &[]);
            pass.set_bind_group(1, &self.texture_bind_group, &[]);
            pass.set_bind_group(2, &self.sampler_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub fn initialize_canvas() -> Vec<web_sys::HtmlCanvasElement> {
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
    click: u32,
    _padding: u32, // in wgsl all fields will be aligned to 4, 8, 16 bytes
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

fn initialize_mouse_handler(window: &web_sys::Window, mouse: Rc<RefCell<MouseUniform>>) {
    let closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let mut mouse = mouse.borrow_mut();
        mouse.pos[0] = event.client_x() as u32;
        mouse.pos[1] = event.client_y() as u32;
        mouse.click = event.buttons() as u32;
    }) as Box<dyn FnMut(_)>);

    window
        .add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())
        .unwrap();

    window
        .add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())
        .unwrap();

    window
        .add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref())
        .unwrap();

    closure.forget();
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
