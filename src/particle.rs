use metal::*;
use crate::{MetalContext};
use crate::shader_types::Particle;

pub struct SimulateParticles {
    pub buffer: Buffer,
    pub num_particles: usize,
    command_queue: CommandQueue,
}

impl SimulateParticles {
    pub fn new(num_particles: usize, metal_context: &MetalContext) -> Self {
        Self {
            buffer: Self::add_particles(num_particles, metal_context),
            num_particles,
            command_queue: metal_context.device.new_command_queue()
        }
    }

    fn add_particles(num_particles: usize, metal_context: &MetalContext) -> Buffer {
        let size = [500.; 3];
        let mut particles: Vec<Particle> = vec![];
        let [mut x, mut y, mut z] = [0.0; 3];

        x = 0.2 * size[0];
        y = 0.2 * size[2];
        for _ in 0..num_particles {
            x += 16.;
            if x > 0.8 * size[0] {
                x = 0.2 * size[0];
                z += 16.;
                if z > 0.8 * size[2] {
                    z = 0.2 * size[2];
                    y += 16.;
                }
            }
            particles.push(Particle { position: [x + 0., y + 0., z + 0.]});
        }
        metal_context.device.new_buffer_with_data(particles.as_ptr().cast(), (size_of::<Particle>() * num_particles) as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn update(&mut self) {
        let command_buffer = self.command_queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();

        command_encoder.set_buffer(0, Some(&self.buffer), 0);
        command_encoder.dispatch_threads(MTLSize::new(self.num_particles as NSUInteger, 1, 1), MTLSize::new(0, 0, 0));

        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}
