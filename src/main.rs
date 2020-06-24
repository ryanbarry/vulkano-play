use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::{ClearValue, Format};
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::{Dimensions, ImageUsage, StorageImage};
use vulkano::instance::{Instance, PhysicalDevice, RawInstanceExtensions};
use vulkano::pipeline::{viewport::Viewport, ComputePipeline, GraphicsPipeline};
use vulkano::swapchain::{ColorSpace, PresentMode, Surface, SurfaceTransform, Swapchain};
use vulkano::sync::GpuFuture;
use vulkano::VulkanObject;

use image::{ImageBuffer, Rgba};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

void main() {
  vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
  vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

  vec2 z = vec2(0.0, 0.0);
  float i;
  for (i=0.0; i < 1.0; i += 0.005) {
    z = vec2(
      z.x*z.x - z.y*z.y + c.x,
      z.y*z.x + z.x*z.y + c.y
    );

    if (length(z) > 4.0) {
      break;
    }
  }

  vec4 to_write = vec4(vec3(i), 1.0);
  imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}
"
    }
}

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

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

    println!("using physical device: {}", physical.name());

    let caps_surface = surface
        .capabilities(physical)
        .expect("failed to get the surface's capabilities");
    let surface_dims = caps_surface.current_extent.unwrap();
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

    // BEGIN: mandelbrot "offline" render
    let src_data = 0..64;
    let src_buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        {
            let mut bu = BufferUsage::transfer_source();
            bu.transfer_destination = true;
            bu.storage_buffer = true;
            bu
        },
        false,
        src_data,
    )
    .expect("failed to create source buffer");

    let dst_data = (0..64).map(|_| 0);
    let dst_buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        {
            let mut bu = BufferUsage::transfer_destination();
            bu.transfer_source = true;
            bu.storage_buffer = true;
            bu
        },
        false,
        dst_data,
    )
    .expect("failed to create destination buffer");

    let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    builder
        .copy_buffer(src_buf.clone(), dst_buf.clone())
        .expect("failed to add copy buffer command");
    let command_buffer = builder.build().expect("failed to build command buffer");

    let finished = command_buffer
        .execute(queue.clone())
        .expect("failed to execute");

    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let src_data_after = src_buf
        .read()
        .expect("failed to read source data after execution");
    let dst_data_after = dst_buf
        .read()
        .expect("failed to read dest data after execution");
    assert_eq!(&*src_data_after, &*dst_data_after);

    let image = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d {
            width: 1024,
            height: 1024,
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family()),
    )
    .unwrap();

    // doing compute on image to visualize the mandelbrot set
    let mandelbrot_shader = cs::Shader::load(device.clone()).expect("failed to create shader");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &mandelbrot_shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_image(image.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    // done with mandelbrot

    let imgoutbuf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::transfer_destination()
        },
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create the image buffer");

    let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    builder
        .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 0.0, 1.0]))
        .unwrap()
        .dispatch(
            [1024 / 8, 1024 / 8, 1],
            compute_pipeline.clone(),
            set.clone(),
            (),
        )
        .unwrap()
        .copy_image_to_buffer(image.clone(), imgoutbuf.clone())
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let imgoutcontents = imgoutbuf.read().unwrap();
    let imgout = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &imgoutcontents[..]).unwrap();
    imgout.save("gpu-image.png").unwrap();

    // END: mandelbrot "offline" render

    // begin the actual real-time rendering

    let vertex1 = Vertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = Vertex {
        position: [0.0, 0.5],
    };
    let vertex3 = Vertex {
        position: [0.5, -0.25],
    };

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::none()
        },
        false,
        vec![vertex1, vertex2, vertex3].into_iter(),
    )
    .unwrap();

    mod vs {
        vulkano_shaders::shader! {
                    ty: "vertex",
                    src: "
#version 450

layout(location = 0) in vec2 position;
layout(push_constant) uniform PushConstants {
  mat2 rot;
} push_constants;

void main() {
  vec2 rotated = position * push_constants.rot;
  gl_Position = vec4(rotated, 0.0, 1.0);
}
"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
  f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"
        }
    }

    let vs = vs::Shader::load(device.clone()).expect("failed to create vertex shader");
    let fs = fs::Shader::load(device.clone()).expect("failed to create fragment shader");

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
            //.triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
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

    let mut theta = 0f32;
    'running: loop {
        theta = theta + 2.0 * std::f32::consts::PI / 720.0;
        let rot = [[theta.cos(), theta.sin()], [-theta.sin(), theta.cos()]];
        let push_constants = vs::ty::PushConstants { rot: rot };

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
                _ => {}
            }
        }
        let (acqd_swch_img, should_recreate, acquire_future) =
            vulkano::swapchain::acquire_next_image(
                swapchain.clone(),
                Some(::std::time::Duration::new(0, 1_000_000_000u32 / 120)),
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
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        acquire_future
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), acqd_swch_img)
            .then_signal_fence_and_flush()
            .unwrap();

        ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
    }
}
