use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, PhysicalDevice, RawInstanceExtensions};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::{ColorSpace, PresentMode, Surface, SurfaceTransform, Swapchain};
use vulkano::sync::GpuFuture;
use vulkano::VulkanObject;

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
    uv: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position, color, uv);

fn main() {
    let ball_img = image::load_from_memory_with_format(
        include_bytes!("../imgs/ballGrey_09.png"),
        image::ImageFormat::Png,
    )
    .expect("failed to load image")
    .to_rgba();
    let ball_img_dims = vulkano::image::Dimensions::Dim2d {
        width: ball_img.dimensions().0,
        height: ball_img.dimensions().1,
    };
    let ball_img_data = ball_img.into_raw().clone();

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

    let ball_verts: Vec<Vertex> = vec![
        Vertex {
            position: [-0.25, -0.25],
            color: [1.0, 0.3, 0.0, 1.0],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [-0.25, 0.25],
            color: [1.0, 0.3, 0.0, 1.0],
            uv: [0.0, 1.0],
        },
        Vertex {
            position: [0.25, 0.25],
            color: [1.0, 0.0, 0.4, 1.0],
            uv: [1.0, 1.0],
        },
        Vertex {
            position: [-0.25, -0.25],
            color: [1.0, 0.3, 0.0, 1.0],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [0.25, 0.25],
            color: [1.0, 0.0, 0.4, 1.0],
            uv: [1.0, 1.0],
        },
        Vertex {
            position: [0.25, -0.25],
            color: [1.0, 0.0, 0.4, 1.0],
            uv: [1.0, 0.0],
        },
    ];
    let ball_vbuff = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::none()
        },
        false,
        ball_verts.into_iter(),
    )
    .unwrap();

    mod vs {
        vulkano_shaders::shader! {
                    ty: "vertex",
                    src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;
layout(push_constant) uniform PushConstants {
  mat2 rot;
  vec2 translation;
  float scale;
} push_constants;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

void main() {
  vec2 scaled = position * push_constants.scale;
  vec2 rotated = scaled * push_constants.rot;
  vec2 positioned = rotated + push_constants.translation;
  gl_Position = vec4(positioned.x*600.0/800.0, positioned.y, 0.0, 1.0);
  v_color = color;
  v_uv = uv;
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
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
  f_color = texture(tex, v_uv);
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
layout(location = 1) in vec2 v_uv;

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

    let (ball_tex, ball_tex_fut) = vulkano::image::ImmutableImage::from_iter(
        ball_img_data.into_iter(),
        ball_img_dims,
        vulkano::format::Format::R8G8B8A8Srgb,
        queue.clone(),
    )
    .expect("failed to create immutable image future");

    let sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    .expect("failed to create sampler");

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .blend_alpha_blending()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
    let ball_ds = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_sampled_image(ball_tex.clone(), sampler.clone())
            .unwrap()
            .build()
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

    let mut prev_presentations = ball_tex_fut.boxed();
    let mut presentations_since_cleanup = 0;
    let mut debug_on = false;
    let mut pos = [0f32; 2];
    let mut v_x = 0.005f32;
    let mut v_y = 0.0071f32;
    'running: loop {
        pos[0] += v_x;
        const MAX_X: f32 = 800.0 / 600.0 - ((0.2 * 0.5) / 2.0);
        const MIN_X: f32 = -800.0 / 600.0 + ((0.2 * 0.5) / 2.0);
        if pos[0] > MAX_X {
            v_x = -v_x;
            pos[0] = MAX_X;
        } else if pos[0] < MIN_X {
            v_x = -v_x;
            pos[0] = MIN_X;
        }
        pos[1] += v_y;
        const MIN_Y: f32 = 1.0 - ((0.2 * 0.5) / 2.0);
        const MAX_Y: f32 = -1.0 + ((0.2 * 0.5) / 2.0);
        if pos[1] > MIN_Y {
            v_y = -v_y;
            pos[1] = MIN_Y;
        } else if pos[1] < MAX_Y {
            v_y = -v_y;
            pos[1] = MAX_Y;
        }

        let ball_pcs = vs::ty::PushConstants {
            rot: [[1.0, 0.0], [0.0, 1.0]],
            translation: pos,
            scale: 0.2,
        };

        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. }
                | sdl2::event::Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::Escape),
                    ..
                } => break 'running,
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
                ball_vbuff.clone(),
                ball_ds.clone(),
                ball_pcs,
            )
            .unwrap();

        if debug_on {
            builder
                .draw(
                    dbg_pipeline.clone(),
                    &dynamic_state,
                    ball_vbuff.clone(),
                    (),
                    ball_pcs,
                )
                .unwrap();
        }

        builder.end_render_pass().unwrap();

        let command_buffer = builder.build().unwrap();

        let pres_fut = acquire_future
            .join(prev_presentations)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
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
