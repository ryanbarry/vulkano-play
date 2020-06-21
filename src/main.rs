use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::{ClearValue, Format};
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::{Instance, PhysicalDevice, RawInstanceExtensions};
use vulkano::pipeline::{viewport::Viewport, ComputePipeline, GraphicsPipeline};
use vulkano::swapchain::Surface;
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
    let surface =
        unsafe { Surface::from_raw_surface(instance.clone(), h_surface, window.context()) };

    let physical = PhysicalDevice::enumerate(&instance)
        .find(|&pd| pd.ty() == vulkano::instance::PhysicalDeviceType::DiscreteGpu)
        .expect("failed to find a discrete gpu");

    println!("using physical device: {}", physical.name());

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        Device::new(
            physical,
            &Features::none(),
            &DeviceExtensions {
                khr_storage_buffer_storage_class: true, // this is required for the compute shader buffer
                ..DeviceExtensions::none()
            },
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

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

void main() {
  gl_Position = vec4(position, 0.0, 1.0);
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
                                                                        format: Format::R8G8B8A8Unorm,
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
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0..1.0,
        }]),
        ..DynamicState::none()
    };

    let framebuffer = Arc::new(
        Framebuffer::start(render_pass.clone())
            .add(image.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let renderoutbuf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::transfer_destination()
        },
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create the image buffer");

    let mut builder =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
    builder
        .begin_render_pass(
            framebuffer.clone(),
            false,
            vec![[0.0, 0.0, 1.0, 1.0].into()],
        )
        .unwrap()
        .draw(
            pipeline.clone(),
            &dynamic_state,
            vertex_buffer.clone(),
            (),
            (),
        )
        .unwrap()
        .end_render_pass()
        .unwrap()
        .copy_image_to_buffer(image.clone(), renderoutbuf.clone())
        .unwrap();

    let command_buffer = builder.build().unwrap();
    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let rendercontents = renderoutbuf.read().unwrap();
    let rendered = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &rendercontents[..]).unwrap();
    rendered.save("gpu-triangle.png").unwrap();

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. }
                | sdl2::event::Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }
        ::std::thread::sleep(::std::time::Duration::new(0, 1_000_000_000u32 / 60));
    }
}
