use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{Dimensions, StorageImage};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

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

fn main() {
    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

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
        .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0]))
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
}
