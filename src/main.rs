use std::sync::Arc;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
  uint data[];
} buf;

void main() {
uint idx = gl_GlobalInvocationID.x;
buf.data[idx] *= 12;
}"
    }
}

fn main() {
    println!("Hello, world!");

    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    for p in PhysicalDevice::enumerate(&instance) {
        println!("device: {:?}", p.name());
        for m in p.memory_types() {
            println!("  memory type: {:?}", m);
        }
    }

    let physical = PhysicalDevice::enumerate(&instance)
        .nth(0)
        .expect("failed to get 2nd physical device");

    println!("physical: {:?}", physical.name());

    for family in physical.queue_families() {
        println!(
            "found a queue family with {:?} queue(s), supports graphics? {:?}",
            family.queues_count(),
            family.supports_graphics()
        );
    }

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        Device::new(
            physical,
            &Features::none(),
            &DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::none()
            },
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let data = 12;
    let buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, data)
        .expect("failed to create buffer");

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

    let data_iter = 0..65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::none()
        },
        false,
        data_iter,
    )
    .expect("failed to create buffer for simd demo");

    let shader = cs::Shader::load(device.clone()).expect("failed to create shader");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
    builder
        .dispatch([1024, 1, 1], compute_pipeline.clone(), set.clone(), ())
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let finished = command_buffer
        .execute(queue.clone())
        .expect("failed to execute simd demo");
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("whoa");
}
