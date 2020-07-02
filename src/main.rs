use std::sync::Arc;

use vulkano::buffer::BufferUsage;
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
    let ball_img = image::open("imgs/ballGrey_09.png")
        .expect("failed to open image")
        .to_rgba();

    let padl_img = image::open("imgs/paddle_01.png")
        .expect("failed to open image")
        .to_rgba();

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
        .allow_highdpi()
        .vulkan()
        .build()
        .expect("failed to create window");

    sdl_context.mouse().capture(true);
    sdl_context.mouse().set_relative_mouse_mode(true);
    sdl_context.mouse().show_cursor(false);

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

    let (ball_fut, mut spr_ball) = Sprite::new(queue.clone(), ball_img);
    spr_ball.scale = spr_ball.width as f32 / 600. * 2. * 0.2;
    let (padl_fut, mut spr_padl) = Sprite::new(queue.clone(), padl_img);
    spr_padl.pos_y = 0.8;
    spr_padl.scale = spr_padl.width as f32 / 600. * 2. * 0.2;

    let padl_desc_set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_sampled_image(spr_padl.texture.clone(), sampler.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let ball_desc_set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_sampled_image(spr_ball.texture.clone(), sampler.clone())
            .unwrap()
            .build()
            .expect("failed to create descriptor set with sampled image"),
    );

    let spr_fut = ball_fut.join(padl_fut);

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

    let mut prev_presentations = spr_fut.boxed();
    let mut presentations_since_cleanup = 0;
    let mut debug_on = false;
    let mut v_x = 0.005f32;
    let mut v_y = 0.0071f32;
    'running: loop {
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
                sdl2::event::Event::MouseMotion {
                    xrel: mousex_rel, ..
                } => {
                    spr_padl.pos_x += (mousex_rel as f32) / 800.0;
                }
                _ => {}
            }
        }

        spr_ball.pos_x += v_x;
        const MAX_X: f32 = 800.0 / 600.0 - ((0.2 * 0.5) / 2.0);
        const MIN_X: f32 = -800.0 / 600.0 + ((0.2 * 0.5) / 2.0);
        if spr_ball.pos_x > MAX_X {
            v_x = -v_x;
            spr_ball.pos_x = MAX_X;
        } else if spr_ball.pos_x < MIN_X {
            v_x = -v_x;
            spr_ball.pos_x = MIN_X;
        }
        spr_ball.pos_y += v_y;
        const MIN_Y: f32 = 1.0 - ((0.2 * 0.5) / 2.0);
        const MAX_Y: f32 = -1.0 + ((0.2 * 0.5) / 2.0);
        if spr_ball.pos_y > MIN_Y {
            v_y = -v_y;
            spr_ball.pos_y = MIN_Y;
        } else if spr_ball.pos_y < MAX_Y {
            v_y = -v_y;
            spr_ball.pos_y = MAX_Y;
        }

        let ball_pcs = vs::ty::PushConstants {
            rot: spr_ball.rotat,
            translation: [spr_ball.pos_x, spr_ball.pos_y],
            scale: spr_ball.scale,
        };
        let paddle_pcs = vs::ty::PushConstants {
            rot: spr_padl.rotat,
            scale: spr_padl.scale,
            translation: [spr_padl.pos_x, spr_padl.pos_y],
        };

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
                vec![[0.3, 0.6, 0.3, 1.0].into()],
            )
            .unwrap()
            .draw(
                pipeline.clone(),
                &dynamic_state,
                spr_ball.vbuff.clone(),
                ball_desc_set.clone(),
                ball_pcs,
            )
            .unwrap()
            .draw(
                pipeline.clone(),
                &dynamic_state,
                spr_padl.vbuff.clone(),
                padl_desc_set.clone(),
                paddle_pcs,
            )
            .unwrap();

        if debug_on {
            builder
                .draw(
                    dbg_pipeline.clone(),
                    &dynamic_state,
                    spr_ball.vbuff.clone(),
                    (),
                    ball_pcs,
                )
                .unwrap()
                .draw(
                    dbg_pipeline.clone(),
                    &dynamic_state,
                    spr_padl.vbuff.clone(),
                    (),
                    paddle_pcs,
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

struct Sprite {
    pos_x: f32,
    pos_y: f32,
    scale: f32,
    rotat: [[f32; 2]; 2],
    height: u32,
    width: u32,
    vbuff: Arc<vulkano::buffer::immutable::ImmutableBuffer<[Vertex]>>,
    texture: Arc<vulkano::image::ImmutableImage<vulkano::format::R8G8B8A8Srgb>>,
}

impl Sprite {
    fn new(
        queue: Arc<vulkano::device::Queue>,
        image_data: image::RgbaImage,
    ) -> (Box<dyn vulkano::sync::GpuFuture>, Sprite) {
        let (img_w, img_h) = image_data.dimensions();
        let vko_dims = vulkano::image::Dimensions::Dim2d {
            width: img_w,
            height: img_h,
        };
        let (tex, tex_fut) = vulkano::image::ImmutableImage::from_iter(
            image_data.into_raw().into_iter(),
            vko_dims,
            vulkano::format::R8G8B8A8Srgb,
            queue.clone(),
        )
        .expect("failed to create vulkan image");

        let (vert_halfwidth, vert_halfheight) = {
            if img_w == img_h {
                (0.5, 0.5)
            } else if img_w > img_h {
                let vhh = (img_h as f32 / img_w as f32) / 2.0;
                (0.5, vhh)
            } else {
                let vhw = (img_w as f32 / img_h as f32) / 2.0;
                (vhw, 0.5)
            }
        };

        let verts = vec![
            Vertex {
                position: [-vert_halfwidth, -vert_halfheight],
                color: [0.0, 0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [-vert_halfwidth, vert_halfheight],
                color: [0.0, 0.0, 0.0, 1.0],
                uv: [0.0, 1.0],
            },
            Vertex {
                position: [vert_halfwidth, vert_halfheight],
                color: [0.0, 0.0, 0.0, 1.0],
                uv: [1.0, 1.0],
            },
            Vertex {
                position: [vert_halfwidth, vert_halfheight],
                color: [0.0, 0.0, 0.0, 1.0],
                uv: [1.0, 1.0],
            },
            Vertex {
                position: [-vert_halfwidth, -vert_halfheight],
                color: [0.0, 0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [vert_halfwidth, -vert_halfheight],
                color: [0.0, 0.0, 0.0, 1.0],
                uv: [1.0, 0.0],
            },
        ];

        let (vert_buf, vbuf_fut) = vulkano::buffer::ImmutableBuffer::from_iter(
            verts.into_iter(),
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::none()
            },
            queue,
        )
        .expect("failed to create vertex buffer");

        (
            Box::new(tex_fut.join(vbuf_fut)),
            Sprite {
                pos_x: 0.0,
                pos_y: 0.0,
                scale: 1.0,
                height: img_h,
                width: img_w,
                rotat: [[1.0, 0.0], [0.0, 1.0]],
                vbuff: vert_buf,
                texture: tex,
            },
        )
    }
}
