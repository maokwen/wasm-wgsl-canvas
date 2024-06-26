mod logger;
mod res;
mod utils;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsValue;

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

const VERTICES: &[f32] = &[
    -1.0, -1.0, 0.0, 1.0, // bottom-left
    1.0, -1.0, 0.0, 1.0, // bottom-right
    -1.0, 1.0, 0.0, 1.0, // top-left
    -1.0, 1.0, 0.0, 1.0, // top-left
    1.0, -1.0, 0.0, 1.0, // bottom-right
    1.0, 1.0, 0.0, 1.0, // top-right
];

const SHADER_DEFINES: &str = r###"
struct Time { frame: u32, elapsed: f32, delta: f32 }
struct Mouse { pos: vec2<u32>, click: u32 }
struct Canvas { resolution: vec2<u32> }

@group(0) @binding(0)   var<uniform> _time: Time;
@group(0) @binding(1)   var<uniform> _mouse: Mouse;
@group(0) @binding(2)   var<uniform> _canvas: Canvas;
"###;

const SHADER_VERT_DEFAULT: &str = r###"
struct VertexInput {
    @location(0) position: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    in: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    let x = f32(in.position.x);
    let y = f32(in.position.y);

    let elapsed = _time.elapsed;
    let rx = (cos(elapsed) * x) - (sin(elapsed) * y) + f32(_mouse.pos.x) / 1000.0;
    let ry = (cos(elapsed) * y) + (sin(elapsed) * x) + f32(_mouse.pos.y) / 1000.0;
    
    out.clip_position = vec4<f32>(in.position);
    out.position = vec2<f32>(rx, ry);
    return out;
}
"###;

const SHADER_FRAG_DEFAULT: &str = r###"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.position, 0.5, 1.0);
}
"###;

#[wasm_bindgen]
pub async fn run() -> Result<(), JsValue> {
    logger::init_logger();
    utils::set_panic_hook();

    let canvas_list = initialize_canvas();

    if canvas_list.is_empty() {
        log::info!("No canvas found");
        return Ok(());
    }

    let mut context_list: Vec<Context> = vec![];

    for canvas in canvas_list {
        let context = Context::new(canvas)
            .await
            .expect("Failed to initialize context");
        context_list.push(context);
    }

    let time_begin = web_time::Instant::now();
    let mut frame: u32 = 0;
    let mut time_last = web_time::Instant::now();

    let mut draw_frame = move || {
        for context in &mut context_list {
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
    canvas_buffer: wgpu::Buffer,
    texture_bindgroup_layout: wgpu::BindGroupLayout,
    channel0_texture: wgpu::Texture,
    channel1_texture: wgpu::Texture,
    channel0_view: wgpu::TextureView,
    channel1_view: wgpu::TextureView,
    time_uniform: TimeUniform,
    mouse_uniform: Rc<RefCell<MouseUniform>>,
    canvas_uniform: CanvasUniform,
    vertices: Vec<f32>,
    vertex_buffer: wgpu::Buffer,
    shader: wgpu::ShaderModule,
    refresh_observer: web_sys::MutationObserver,
    refresh_flag: Rc<RefCell<bool>>,
    resize_observer: web_sys::MutationObserver,
    resize_flag: Rc<RefCell<bool>>,
}

impl Context<'_> {
    async fn new(canvas: web_sys::HtmlCanvasElement) -> anyhow::Result<Self> {
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

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: 512,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (vertices, shader_vert, shader_frag) = get_canvas_data(&canvas)?;
        let shader_desc = wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                format!("{SHADER_DEFINES}{shader_vert}{shader_frag}").into(),
            ),
        };
        let shader = device.create_shader_module(shader_desc.clone());

        let (pipeline_layout, perframe_layout, texture_layout, sampler_layout) =
            initialize_pipeline_layout(&device);
        let (perframe_bind_group, time_buffer, mouse_buffer, canvas_buffer) =
            initialize_bind_group_0(&device, &perframe_layout);
        let (texture_bind_group, channel0_texture, channel0_view, channel1_texture, channel1_view) =
            initialize_bind_group_texture(&device, &texture_layout);
        let sampler_bind_group = initialize_bind_group_sampler(&device, &sampler_layout);

        let render_pipeline =
            Context::setup_pipeline(&device, &pipeline_layout, config.format, &shader);

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

        let canvas_uniform = CanvasUniform {
            resolution: [config.width, config.height],
        };
        queue.write_buffer(
            &canvas_buffer,
            0,
            bytemuck::cast_slice(&[canvas_uniform]),
        );

        initialize_mouse_handler(&window(), mouse_uniform.clone());

        let refresh_flag = Rc::new(RefCell::new(false));
        let refresh_observer =
            initialize_datachange_handler(&window(), &canvas, refresh_flag.clone());

        let resize_flag = Rc::new(RefCell::new(false));
        let resize_observer =
            initialize_sizechange_handler(&window(), &canvas, resize_flag.clone());

        Ok(Context {
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
            canvas_uniform,
            vertices,
            vertex_buffer,
            shader,
            refresh_observer,
            refresh_flag,
            canvas_buffer,
            resize_observer,
            resize_flag,
        })
    }

    fn setup_pipeline(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        shader: &wgpu::ShaderModule,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: (mem::size_of::<f32>() * 4) as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x4,
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions {
                    ..Default::default()
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: format,
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
                cull_mode: None,
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
        })
    }

    fn config_data(&mut self) {
        match get_canvas_data(&self.canvas) {
            Ok((vertices, shader_vert, shader_frag)) => {
                let shader_desc = wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(
                        format!("{SHADER_DEFINES}{shader_vert}{shader_frag}").into(),
                    ),
                };

                let shader = self.device.create_shader_module(shader_desc.clone());

                self.render_pipeline = Context::setup_pipeline(
                    &self.device,
                    &self.pipeline_layout,
                    self.config.format,
                    &shader,
                );
            }
            Err(e) => log::error!("Failed to get canvas data: {:?}", e),
        }
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
        self.canvas_uniform.resolution = [width, height];
        self.queue.write_buffer(
            &self.canvas_buffer,
            0,
            bytemuck::cast_slice(&[self.canvas_uniform]),
        );
    }

    fn input(&self) {}

    fn update(&mut self) {
        if *self.refresh_flag.borrow() {
            self.config_data();
            self.refresh_flag.replace(false);
        }
        if *self.resize_flag.borrow() {
            self.resize();
            self.resize_flag.replace(false);
        }
    }

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
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(self.vertices.as_slice()),
            );

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
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..(self.vertices.len() as u32), 0..1);
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
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
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
    // binding(2) var<uniform> _canvas: Canvas { resolution }
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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CanvasUniform {
    resolution: [u32; 2],
}

pub fn initialize_bind_group_0(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    // binding(0) var<uniform> _time: Time { frame, elapsed }
    // binding(1) var<uniform> _mouse: Mouse { pos, click }
    // binding(2) var<uniform> _canvas: Canvas { resolution }

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

    let canvas_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("_canvas Buffer"),
        size: mem::size_of::<CanvasUniform>() as wgpu::BufferAddress,
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: canvas_buffer.as_entire_binding(),
            },
        ],
    });

    (bind_group, time_buffer, mouse_buffer, canvas_buffer)
}

fn initialize_mouse_handler(window: &web_sys::Window, mouse: Rc<RefCell<MouseUniform>>) {
    let closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let mut mouse = mouse.borrow_mut();
        mouse.pos[0] = event.client_x() as u32;
        mouse.pos[1] = event.client_y() as u32;
        mouse.click = event.buttons() as u32;
    }) as Box<dyn FnMut(_)>);

    window.add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref());

    window.add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref());

    window.add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref());

    closure.forget(); // fuck
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

fn initialize_datachange_handler(
    window: &web_sys::Window,
    canvas: &web_sys::HtmlCanvasElement,
    refresh_flag: Rc<RefCell<bool>>,
) -> web_sys::MutationObserver {
    let observer_func =
        Closure::<dyn FnMut(JsValue, JsValue)>::new(move |_records: JsValue, _observer| {
            log::info!("Canvas data changed");
            refresh_flag.replace(true);
        });
    let mutation_observer =
        web_sys::MutationObserver::new(observer_func.as_ref().unchecked_ref()).unwrap();

    let attribute_filter = JsValue::from(
        ["data-frag", "data-vert", "data-vertices"]
            .iter()
            .copied()
            .map(JsValue::from)
            .collect::<js_sys::Array>(),
    );
    let mut mutation_observer_init = web_sys::MutationObserverInit::new();
    mutation_observer_init.attribute_filter(&attribute_filter);
    mutation_observer.observe_with_options(canvas, &mutation_observer_init);

    observer_func.forget(); // fuck

    mutation_observer
}

fn initialize_sizechange_handler(
    window: &web_sys::Window,
    canvas: &web_sys::HtmlCanvasElement,
    resize_flag: Rc<RefCell<bool>>,
) -> web_sys::MutationObserver {
    let observer_func =
        Closure::<dyn FnMut(JsValue, JsValue)>::new(move |_records: JsValue, _observer| {
            log::info!("Canvas size changed");
            resize_flag.replace(true);
        });
    let mutation_observer =
        web_sys::MutationObserver::new(observer_func.as_ref().unchecked_ref()).unwrap();

    let attribute_filter = JsValue::from(
        ["width", "height"]
            .iter()
            .copied()
            .map(JsValue::from)
            .collect::<js_sys::Array>(),
    );
    let mut mutation_observer_init = web_sys::MutationObserverInit::new();
    mutation_observer_init.attribute_filter(&attribute_filter);
    mutation_observer.observe_with_options(canvas, &mutation_observer_init);

    observer_func.forget(); // fuck

    mutation_observer
}

pub fn get_canvas_data(
    canvas: &web_sys::HtmlCanvasElement,
) -> anyhow::Result<(Vec<f32>, String, String)> {
    let vertices: Vec<f32>;
    let data_vertices = canvas.get_attribute("data-vertices");
    if data_vertices.as_ref().is_some_and(|x| !x.is_empty()) {
        vertices = data_vertices
            .unwrap()
            .split(",")
            .map(|x| x.parse::<f32>().expect("Failed to parse data-vertices"))
            .collect::<Vec<f32>>();
    } else {
        vertices = VERTICES.to_vec();
    }

    let mut shader_vert: String;
    let data_shader_vert = canvas.get_attribute("data-vert");
    if data_shader_vert.as_ref().is_some_and(|x| !x.is_empty()) {
        shader_vert = data_shader_vert.unwrap();
    } else {
        shader_vert = SHADER_VERT_DEFAULT.to_owned();
    }

    let mut shader_frag: String;
    let data_shader_frag = canvas.get_attribute("data-frag");
    if data_shader_frag.as_ref().is_some_and(|x| !x.is_empty()) {
        shader_frag = data_shader_frag.unwrap();
    } else {
        shader_frag = SHADER_FRAG_DEFAULT.to_owned();
    }

    Ok((vertices, shader_vert, shader_frag))
}
