use crate::MetalContext;
use crate::shader_types::{ComputeArguments, Particle, simd_float3};
use metal::*;
use rand::prelude::*;
use std::env;

pub struct ComputePipeline {
    name: String,
    compute_pipeline: ComputePipelineState,
}

impl ComputePipeline {
    pub fn new(name: &str, metal_context: &MetalContext) -> Self {
        let function = metal_context
            .library
            .get_function(name, None)
            .expect("function not found");
        let descriptor = ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        descriptor.set_label(name);

        Self {
            name: name.to_string(),
            compute_pipeline: metal_context
                .device
                .new_compute_pipeline_state(&descriptor)
                .expect("unable to create compute pipeline"),
        }
    }

    pub fn dispatch(&self, encoder: &ComputeCommandEncoderRef, num_threads: usize) {
        encoder.set_compute_pipeline_state(&self.compute_pipeline);
        encoder.dispatch_threads(
            MTLSize::new(num_threads as NSUInteger, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }
}

macro_rules! compute_args {
    ($args:expr, $elem:ident) => {
        unsafe { *($args.contents() as *mut ComputeArguments) }.$elem
    };
    ($args:expr, $elem:ident, $val:expr) => {
        unsafe { ($args.contents() as *mut ComputeArguments).as_mut() }.unwrap().$elem = $val
    };
}

pub struct SimulateParticles {
    pub buffer: Buffer,
    pub arg_buffer: Buffer,
    pub num_particles: usize,
    command_queue: CommandQueue,
    non_pressure_forces: ComputePipeline,
    compute_densities_and_factors: ComputePipeline,
    adapt_timestep: ComputePipeline,
    correct_density_error: ComputePipeline,
    update_positions: ComputePipeline,
    update_velocities: ComputePipeline,
    compute_pressure_accel: ComputePipeline,
    compute_density_adv: ComputePipeline,
    update_velocities_from_pressure: ComputePipeline,
    run: bool,
}

impl SimulateParticles {
    pub fn new(num_particles: usize, metal_context: &MetalContext) -> Self {
        let particle_radius = 0.025f32;
        let buffer = Self::add_particles(num_particles, particle_radius,  metal_context);
        let command_queue = metal_context.device.new_command_queue();
        let compute_densities_and_factors =
            ComputePipeline::new("compute_densities_and_factors", metal_context);


        let compute_arguments = ComputeArguments {
            num_particles: num_particles as i32,
            particle_radius,
            volume: 0.8 * (particle_radius * 2.).powi(3),
            kernel_radius: 4. * particle_radius,
            time_step: unsafe { std::mem::transmute(0.0005f32) },
            rest_density: 1000.0f32,
            ..Default::default()
        };
        let arg_buffer = metal_context.device.new_buffer_with_data(
            (&compute_arguments as *const ComputeArguments).cast(),
            size_of::<ComputeArguments>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            buffer,
            arg_buffer,
            num_particles,
            command_queue,
            non_pressure_forces: ComputePipeline::new("non_pressure_forces", metal_context),
            compute_densities_and_factors,
            adapt_timestep: ComputePipeline::new("adapt_timestep", metal_context),
            correct_density_error: ComputePipeline::new("correct_density_error", metal_context),
            update_positions: ComputePipeline::new("update_positions", metal_context),
            // correct_divergence_error: ComputePipeline::new(
            //     "correct_divergence_error",
            //     metal_context,
            // ),
            update_velocities: ComputePipeline::new("update_velocities", metal_context),
            compute_pressure_accel: ComputePipeline::new("compute_pressure_accel", metal_context),
            compute_density_adv: ComputePipeline::new("compute_density_adv", metal_context),
            update_velocities_from_pressure: ComputePipeline::new(
                "update_velocities_from_pressure",
                metal_context,
            ),
            run: true,
        }
    }

    fn add_particles(num_particles: usize, radius: f32, metal_context: &MetalContext) -> Buffer {
        let size = [0.6f32, 1f32, 0.6f32];
        let mut particles: Vec<Particle> = vec![];

        let start_x = 0.2 * size[0];
        let start_y = 0.2 * size[0];
        let start_z = 0.2 * size[2];
        let dist_between_particles = radius * 0.9; //2.25f32 * radius;
        let rows = 15;
        let cols = 15;
        for i in 0..num_particles {
            let x = i / (rows * cols);
            let ind = i - x * rows * cols;
            let y = ind / rows;
            let z = ind % rows;
            particles.push(Particle {
                position: simd_float3::new([
                    dist_between_particles * (x as f32) + start_x,
                    dist_between_particles * (y as f32) + start_y,
                    dist_between_particles * (z as f32) + start_z,
                ]),
                ..Default::default()
            });
        }

        metal_context.device.new_buffer_with_data(
            particles.as_ptr().cast(),
            (size_of::<Particle>() * num_particles) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn add_particles2(num_particles: usize, radius: f32, metal_context: &MetalContext) -> Buffer {
        let size = [0.6f32, 20f32, 0.6f32];
        let mut particles: Vec<Particle> = vec![];
        let [mut x, mut y, mut z] = [0.0; 3];

        x = 0.2 * size[0];
        y = 0.2 * size[2];
        let dist_between_particles = radius * 1.; //2.25f32 * radius;
        let offset = 0.; //dist_between_particles * 1.;
        for _ in 0..num_particles {
            x += dist_between_particles;
            if x > 0.8 * size[0] {
                x = 0.2 * size[0];
                z += dist_between_particles;
                if z > 0.8 * size[2] {
                    z = 0.2 * size[2];
                    y += dist_between_particles;
                }
            }

            particles.push(Particle {
                position: simd_float3::new([
                    x + rand::random::<f32>() * offset,
                    y + rand::random::<f32>() * dist_between_particles * offset,
                    z + rand::random::<f32>() * offset,
                ]),
                ..Default::default()
            });
        }
        metal_context.device.new_buffer_with_data(
            particles.as_ptr().cast(),
            (size_of::<Particle>() * num_particles) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn update(&mut self, metal_context: &MetalContext) {
        if !self.run {
            //return;
        }

        let do_density = true;
        let do_divergence = false;

        if env::var("XCODE").is_ok() && !self.run {
            CaptureManager::shared().start_capture_with_scope(&metal_context.capture_scope);
            metal_context.capture_scope.begin_scope()
        }

        self.update_compute_arguments();

        self.dispatch_compute_pipelines(&[&self.compute_densities_and_factors]);

        if do_divergence {
            self.correct_divergence();
        }

        self.dispatch_compute_pipelines(&[&self.non_pressure_forces, &self.adapt_timestep, &self.update_velocities]);

        println!("timestep: {}", unsafe { std::mem::transmute::<u32, f32>(compute_args!(self.arg_buffer, time_step))}) ;
        
        if do_density {
            self.correct_density();
        }

        self.dispatch_compute_pipelines(&[&self.update_positions]);

        self.run = false;
        if env::var("XCODE").is_ok() {
            metal_context.capture_scope.end_scope()
        }
    }

    fn correct_density(&mut self) {
        let mut iterations = 0;
        let error = 10f32;  // percent

        self.dispatch_compute_pipelines(&[&self.compute_density_adv]);

        while {
            let cont = unsafe { (compute_args!(self.arg_buffer, density_error)) >= (error * 0.001f32 * compute_args!(self.arg_buffer, rest_density)) } ;
            println!("density_error: {}", compute_args!(self.arg_buffer, density_error));
            println!("iterations: {}", iterations);
            compute_args!(self.arg_buffer, density_error, 0.);
            (iterations < 2 || cont) && !(iterations > 200)
        } {
            self.dispatch_compute_pipelines(&[&self.compute_pressure_accel, &self.correct_density_error]);
            iterations += 1;
        }

        self.dispatch_compute_pipelines(&[&self.compute_pressure_accel, &self.update_velocities_from_pressure]);
    }

    fn correct_divergence(&mut self) {

    }
    //
    //
    //
    //     while {
    //         let compute_arguments = self.arg_buffer.contents() as *mut ComputeArguments;
    //         unsafe {
    //             let b = ((((*compute_arguments).avg_pred_density)
    //                 - (*compute_arguments).rest_density)
    //                 .abs()
    //                 > 0.01f32)
    //                 || iterations < 2;
    //             if iterations == 1 || iterations == 200 {
    //                 println!(
    //                     "avg_pred_density: {}",
    //                     (*compute_arguments).avg_pred_density
    //                 );
    //             }
    //             (*compute_arguments).avg_pred_density = 0.;
    //             b && iterations < 200 && do_density
    //         }
    //     } {
    //         dispatch_compute(&self.command_queue, |command_encoder| {
    //             command_encoder.set_buffer(0, Some(&self.buffer), 0);
    //             command_encoder.set_buffer(1, Some(&self.arg_buffer), 0);
    //             command_encoder.set_label(format!("Correct Density Error: {iterations}").as_str());
    //             self.compute_density_pred_derivative
    //                 .dispatch(command_encoder, self.num_particles);
    //             self.correct_density_error
    //                 .dispatch(command_encoder, self.num_particles);
    //         });
    //         iterations += 1;
    //     }
    //
    //     dispatch_compute(&self.command_queue, |command_encoder| {
    //         command_encoder.set_buffer(0, Some(&self.buffer), 0);
    //         command_encoder.set_buffer(1, Some(&self.arg_buffer), 0);
    //         self.update_positions
    //             .dispatch(command_encoder, self.num_particles);
    //         self.compute_densities_and_factors
    //             .dispatch(command_encoder, self.num_particles);
    //     });
    //
    //     iterations = 0;
    //     while {
    //         let compute_arguments = self.arg_buffer.contents() as *mut ComputeArguments;
    //         unsafe {
    //             let b = (*compute_arguments).avg_density_derivative.abs() > 100.0 || iterations < 1;
    //             if iterations == 1 || iterations == 200 {
    //                 println!(
    //                     "avg_density_derive: {}",
    //                     (*compute_arguments).avg_density_derivative
    //                 );
    //             }
    //             (*compute_arguments).avg_density_derivative = 0.;
    //             b && iterations < 200 && do_divergence
    //         }
    //     } {
    //         dispatch_compute(&self.command_queue, |command_encoder| {
    //             command_encoder.set_buffer(0, Some(&self.buffer), 0);
    //             command_encoder.set_buffer(1, Some(&self.arg_buffer), 0);
    //             self.compute_density_pred_derivative
    //                 .dispatch(command_encoder, self.num_particles);
    //             self.correct_divergence_error
    //                 .dispatch(command_encoder, self.num_particles);
    //         });
    //         iterations += 1;
    //     }
    //
    //     dispatch_compute(&self.command_queue, |command_encoder| {
    //         command_encoder.set_buffer(0, Some(&self.buffer), 0);
    //         command_encoder.set_buffer(1, Some(&self.arg_buffer), 0);
    //         self.update_velocities
    //             .dispatch(command_encoder, self.num_particles);
    //     });
    //     self.run = false;
    //     if env::var("XCODE").is_ok() {
    //         metal_context.capture_scope.end_scope()
    //     }
    // }

    pub fn update_compute_arguments(&mut self) {
        compute_args!(self.arg_buffer, time_step) = unsafe { std::mem::transmute(0.0005f32) };
        compute_args!(self.arg_buffer, avg_density_derivative) = 0.;
        compute_args!(self.arg_buffer, density_error) = 0.;
    }

    fn dispatch_compute<F>(&self, func: F)
    where
        F: Fn(&ComputeCommandEncoderRef),
    {
        let command_buffer = self.command_queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();
        command_encoder.set_buffer(0, Some(&self.buffer), 0);
        command_encoder.set_buffer(1, Some(&self.arg_buffer), 0);
        func(command_encoder);

        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    fn dispatch_compute_pipelines(&self, pipelines: &[&ComputePipeline]) {
        self.dispatch_compute(|command_encoder| {
            command_encoder.set_label(pipelines.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ").as_str());
            for pipeline in pipelines {
                pipeline.dispatch(command_encoder, self.num_particles);
            }
        })
    }
}
