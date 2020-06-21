// from https://github.com/SamP20/HelloTriangle/blob/c9a3db54b0396d9ab4af9dfafda06cafe88024e8/src/sendable.rs
// required to satisfy Vulkano's requirement that the framebuffer given to begin_render_pass be Send + Sync.

use std::thread;

pub struct Sendable<T> {
    data: T,
    thread: thread::ThreadId,
}

unsafe impl<T> Send for Sendable<T> {}
unsafe impl<T> Sync for Sendable<T> {}

impl<T> Sendable<T> {
    pub fn new(data: T) -> Sendable<T> {
        Sendable {
            data: data,
            thread: thread::current().id(),
        }
    }

    pub fn get(&self) -> Option<&T> {
        if thread::current().id() == self.thread {
            Some(&self.data)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        if thread::current().id() == self.thread {
            Some(&mut self.data)
        } else {
            None
        }
    }
}

impl<T> Drop for Sendable<T> {
    fn drop(&mut self) {
        if thread::current().id() != self.thread {
            panic!("Unsafe drop from a different thead.");
        }
    }
}
