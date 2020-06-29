use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, PhysicalDevice, RawInstanceExtensions};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline};
use vulkano::swapchain::{ColorSpace, PresentMode, Surface, SurfaceTransform, Swapchain};
use vulkano::sync::GpuFuture;
use vulkano::VulkanObject;

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}
vulkano::impl_vertex!(Vertex, position, color);

fn main() {
    let sdl_context = sdl2::init().expect("failed to initialize SDL2");
    println!(
        "using SDL2 version {}, rev {}",
        sdl2::version::version(),
        sdl2::version::revision()
    );
    let mut event_pump = sdl_context
        .event_pump()
        .expect("failed to get SDL event pump");
    let video_subsystem = sdl_context
        .video()
        .expect("failed to get SDL video subsystem");
    let window = video_subsystem
        .window("Vulkano Play", 800, 600)
        .vulkan()
        .build()
        .expect("failed to create window");

    let instance_extensions = window.vulkan_instance_extensions().unwrap();
    let vk_inst_exts = RawInstanceExtensions::new(
        instance_extensions
            .iter()
            .map(|&v| std::ffi::CString::new(v).unwrap()),
    );

    let instance = Instance::new(None, vk_inst_exts, None).expect("failed to create instance");

    let h_surface = window
        .vulkan_create_surface(instance.internal_object())
        .expect("failed to create surface");

    // NOTE: we are giving the surface an empty tuple as the "window" because while SDL2 does not
    // support using the window from multiple threads, Vulkano requires the surface to be
    // Send + Sync through the framebuffer (in turn, through the swapchain). this means, instead of
    // being able to rely on the explicit relationships to keep objects alive, we have to ensure
    // that the window is alive for the entire life of the surface ourself.
    let surface = Arc::new(unsafe { Surface::from_raw_surface(instance.clone(), h_surface, ()) });

    let physical = PhysicalDevice::enumerate(&instance)
        .find(|&pd| pd.ty() == vulkano::instance::PhysicalDeviceType::DiscreteGpu)
        .expect("failed to find a discrete gpu");

    println!("selected physical device: {}", physical.name());

    let caps_surface = surface
        .capabilities(physical)
        .expect("failed to get the surface's capabilities");
    let surface_dims = match caps_surface.current_extent {
        Some(dims_arr) => dims_arr,
        None => {
            let dims_tup = window.vulkan_drawable_size();
            [dims_tup.0, dims_tup.1]
        }
    };
    let surface_alpha = caps_surface
        .supported_composite_alpha
        .iter()
        .next()
        .unwrap();
    let surface_format = caps_surface.supported_formats[0].0;

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        Device::new(
            physical,
            physical.supported_features(),
            &DeviceExtensions {
                khr_storage_buffer_storage_class: true, // this is required for the compute shader buffer
                khr_swapchain: true,
                ..DeviceExtensions::none()
            },
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let (swapchain, swapchain_images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        caps_surface.min_image_count,
        surface_format,
        surface_dims,
        1,
        ImageUsage::color_attachment(),
        &queue,
        SurfaceTransform::Identity,
        surface_alpha,
        PresentMode::Fifo,
        vulkano::swapchain::FullscreenExclusive::AppControlled, // TODO: research this a little
        true,
        ColorSpace::SrgbNonLinear,
    )
    .expect("failed to create swapchain");

    let vertices: Vec<Vertex> = vec![
        Vertex {
            position: [-0.25, -0.25],
            color: [1.0, 0.3, 0.0, 1.0],
        },
        Vertex {
            position: [-0.25, 0.25],
            color: [1.0, 0.3, 0.0, 1.0],
        },
        Vertex {
            position: [0.25, 0.25],
            color: [1.0, 0.0, 0.4, 1.0],
        },
        Vertex {
            position: [-0.25, -0.25],
            color: [1.0, 0.3, 0.0, 1.0],
        },
        Vertex {
            position: [0.25, 0.25],
            color: [1.0, 0.0, 0.4, 1.0],
        },
        Vertex {
            position: [0.25, -0.25],
            color: [1.0, 0.0, 0.4, 1.0],
        },
    ];

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::none()
        },
        false,
        vertices.into_iter(),
    )
    .unwrap();

    mod vs {
        vulkano_shaders::shader! {
                    ty: "vertex",
                    src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(push_constant) uniform PushConstants {
  mat2 rot;
  vec2 translation;
} push_constants;

layout(location = 0) out vec4 v_color;

void main() {
  vec2 rotated = position * push_constants.rot;
  vec2 positioned = rotated + push_constants.translation;
  gl_Position = vec4(positioned.x*600.0/800.0, positioned.y, 0.0, 1.0);
  v_color = color;
}
"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
#version 450

layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 f_color;

void main() {
  f_color = v_color;
}
"
        }
    }

    mod dbg_fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
#version 450

layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 f_color;

void main() {
  f_color = vec4(1.0-v_color.r, 1.0-v_color.g, 1.0-v_color.b, 1.0);
}
"
        }
    }

    let vs = vs::Shader::load(device.clone()).expect("failed to create vertex shader");
    let fs = fs::Shader::load(device.clone()).expect("failed to create fragment shader");
    let dbg_fs =
        dbg_fs::Shader::load(device.clone()).expect("failed to create dbg fragment shader");

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
                                                                attachments: {
                                                                    color: {
                                                                        load: Clear,
                                                                        store: Store,
                                                                        format: swapchain.format(),
                                                                        samples: 1,
                                                                    }
                                                                },
                                                                pass: {
                                                                    color: [color],
                                                                    depth_stencil: {}
                                                                }
    ).unwrap());

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let dbg_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .polygon_mode_line()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(dbg_fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: {
                let dim = swapchain_images[0].dimensions();
                [dim[0] as f32, dim[1] as f32]
            },
            depth_range: 0.0..1.0,
        }]),
        ..DynamicState::none()
    };

    let framebuffers = swapchain_images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();

    let mut prev_presentations = vulkano::sync::now(device.clone()).boxed();
    let mut presentations_since_cleanup = 0;
    let mut theta = 0f32;
    let mut debug_on = false;
    'running: loop {
        theta = theta + 2.0 * std::f32::consts::PI / 1440.0;
        let rot = [[theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]];
        let push_constants = vs::ty::PushConstants {
            rot: rot,
            translation: [0.8 * theta.cos(), 0.8 * theta.sin()],
        };

        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. }
                | sdl2::event::Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::Escape),
                    ..
                } => break 'running,
                sdl2::event::Event::KeyUp {
                    keycode: Some(sdl2::keyboard::Keycode::Space),
                    ..
                } => {
                    println!("rot is {:?}", rot);
                }
                sdl2::event::Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::D),
                    ..
                } => {
                    debug_on = !debug_on;
                }
                _ => {}
            }
        }

        // before acquiring the next image in the swapchain, clean up any past futures
        if presentations_since_cleanup > 5 {
            prev_presentations.cleanup_finished();
            presentations_since_cleanup = 0;
        }

        let (acqd_swch_img, _should_recreate, acquire_future) =
            vulkano::swapchain::acquire_next_image(
                swapchain.clone(),
                //Some(::std::time::Duration::new(0, 1_000_000_000u32 / 120)),
                None, // no timeout
            )
            .unwrap();

        let mut builder =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap();
        builder
            .begin_render_pass(
                framebuffers[acqd_swch_img].clone(),
                false,
                vec![[0.0, 0.0, 1.0, 1.0].into()],
            )
            .unwrap()
            .draw(
                pipeline.clone(),
                &dynamic_state,
                vertex_buffer.clone(),
                (),
                push_constants,
            )
            .unwrap();

        if debug_on {
            builder
                .draw(
                    dbg_pipeline.clone(),
                    &dynamic_state,
                    vertex_buffer.clone(),
                    (),
                    push_constants,
                )
                .unwrap();
        }

        builder.end_render_pass().unwrap();

        let command_buffer = builder.build().unwrap();

        let pres_fut = acquire_future
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .join(prev_presentations)
            .then_swapchain_present(queue.clone(), swapchain.clone(), acqd_swch_img)
            .then_signal_fence_and_flush();

        match pres_fut {
            Ok(future) => {
                presentations_since_cleanup += 1;
                prev_presentations = future.boxed();
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                //recreate_swapchain = true;
                prev_presentations = vulkano::sync::now(device.clone()).boxed();
                presentations_since_cleanup = 0; // drop()ed the prev. value of prev_presentations
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                prev_presentations = vulkano::sync::now(device.clone()).boxed();
                presentations_since_cleanup = 0; // drop()ed the prev. value of prev_presentations
            }
        }
    }
}
