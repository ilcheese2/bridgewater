use crate::MetalContext;
use crate::shader_types::{ComputeArguments, Particle, simd_float3, simd_int3, ParticleLocation};
use metal::*;
use rand::prelude::*;
use std::env;
use std::process::exit;

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

macro_rules! format_particles {
    ($buffer:expr, $($field:ident)+) => {
        unsafe {std::slice::from_raw_parts($buffer.contents() as *mut Particle, $buffer.length() as usize / std::mem::size_of::<Particle>())}
            .iter()
            .map(|p| {
                let mut str = String::new();
                $(
                    str.push_str(format!("{:?} ", *&p.$field).as_str());
                )+
                str
            })
            .collect::<Vec<_>>()
    };
}

pub struct SimulateParticles {
    pub buffer: Buffer,
    pub arg_buffer: Buffer,
    pub cell_buffer: Buffer,
    pub num_particles: usize,
    command_queue: CommandQueue,
    non_pressure_forces: ComputePipeline,
    compute_densities_and_factors: ComputePipeline,
    adapt_timestep: ComputePipeline,
    correct_density_error: ComputePipeline,
    update_positions: ComputePipeline,
    update_velocities: ComputePipeline,
    compute_pressure_accel: ComputePipeline,
    compute_pressure_accel2: ComputePipeline,
    compute_density_adv: ComputePipeline,
    correct_divergence_error: ComputePipeline,
    update_velocities_from_pressure: ComputePipeline,
    compute_cell: ComputePipeline,
    run: bool,
    cell_index_buffer: Buffer,
    runs: i32,
    compute_density_change: ComputePipeline,
    compute_pressure_accel_factor: ComputePipeline,
}

impl SimulateParticles {
    pub fn new(num_particles: usize, metal_context: &MetalContext) -> Self {
        let particle_radius = 0.025;
        let mut size = [0.4f32, 1f32, 0.4f32];

        let buffer = Self::add_particles(num_particles, particle_radius, size,  metal_context);
        size = [2., 2., 2.];
        let command_queue = metal_context.device.new_command_queue();
        let compute_densities_and_factors =
            ComputePipeline::new("compute_densities_and_factors", metal_context);

        let grid_dims = simd_int3::new(size.map(|x|  (x / particle_radius).ceil() as i32 ));
        println!("{:?}", grid_dims);
        let compute_arguments = ComputeArguments {
            num_particles: num_particles as i32,
            particle_radius,
            volume: 0.8 * (particle_radius * 2.).powi(3),
            kernel_radius: 4. * particle_radius,
            time_step: unsafe { std::mem::transmute(0.0001f32) },
            rest_density: 1000.0f32,
            size: simd_float3::new(size),
            grid_dims,
            ..Default::default()
        };
        let arg_buffer = metal_context.device.new_buffer_with_data(
            (&compute_arguments as *const ComputeArguments).cast(),
            size_of::<ComputeArguments>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        //metal_context.library.get_function("compute_densitiy_factor", None)
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
            correct_divergence_error: ComputePipeline::new(
                "correct_divergence_error",
                metal_context,
            ),
            update_velocities: ComputePipeline::new("update_velocities", metal_context),
            compute_pressure_accel: ComputePipeline::new("compute_pressure_accel", metal_context),
            compute_density_adv: ComputePipeline::new("compute_density_adv", metal_context),
            update_velocities_from_pressure: ComputePipeline::new(
                "update_velocities_from_pressure",
                metal_context,
            ),
            compute_pressure_accel2: ComputePipeline::new(
                "compute_pressure_accel2",
                metal_context,
            ),
            compute_pressure_accel_factor: ComputePipeline::new(
                "compute_pressure_accel_factor",
                metal_context,
            ),
            compute_density_change: ComputePipeline::new("compute_density_change", metal_context),
            compute_cell: ComputePipeline::new("compute_cell", metal_context),
            run: true,
            cell_buffer: metal_context.device.new_buffer((num_particles) as u64 * size_of::<ParticleLocation>() as u64, MTLResourceOptions::StorageModeShared),
            cell_index_buffer: metal_context.device.new_buffer((grid_dims[0] * grid_dims[1] * grid_dims[2]) as u64 * size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared),
            runs: 0,
        }
    }

    fn add_particles2(num_particles: usize, radius: f32, size: [f32; 3], metal_context: &MetalContext) -> Buffer {

        let mut particles: Vec<Particle> = vec![];

        let start_x = 0.2 * size[0];
        let start_y = 0.8 * size[0];
        let start_z = 0.2 * size[2];
        let dist_between_particles = radius * 4.; //2.25f32 * radius;
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
        //println!("{:?}", particles.iter().map(|p| p.position.as_slice()).collect::<Vec<_>>());
        metal_context.device.new_buffer_with_data(
            particles.as_ptr().cast(),
            (size_of::<Particle>() * num_particles) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn add_particles(num_particles: usize, radius: f32, size: [f32; 3], metal_context: &MetalContext) -> Buffer {
        let mut particles: Vec<Particle> = vec![];
        let [mut x, mut y, mut z] = [0.0; 3];

        x = 0.2 * size[0];
        y = 0.2 * size[2];
        let dist_between_particles = radius * 5.; //2.25f32 * radius;
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
                    x, //+ rand::random::<f32>() * offset,
                    y, //+ rand::random::<f32>() * dist_between_particles * offset,
                    z //+ rand::random::<f32>() * offset,
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
        self.runs += 1;
        if !self.run {
            //return;
        }

        let do_density = true;
        let do_divergence = true;

        if self.runs > 2 && env::var("XCODE").is_ok()  {
            CaptureManager::shared().start_capture_with_scope(&metal_context.capture_scope);
            metal_context.capture_scope.begin_scope()
        }

        self.update_compute_arguments();
        self.get_neighbors(metal_context);

        self.dispatch_compute_pipelines(&[&self.compute_densities_and_factors]);
        //return;
        //0.0066213896
        //println!("{:?}", format_particles!(self.buffer, factor));
        //println!("{:?}", format_particles!(self.buffer, factor));
        //println!("{:?}", self..iter().map(|p| p.position.as_slice()).collect::<Vec<_>>());
        if do_divergence {
            self.correct_divergence();
        }
        //clear acceleration
        self.dispatch_compute_pipelines(&[&self.non_pressure_forces, &self.adapt_timestep, &self.update_velocities]);

        //println!("timestep: {}", unsafe { std::mem::transmute::<u32, f32>(compute_args!(self.arg_buffer, time_step))}) ;
        
        if do_density {
            self.correct_density(metal_context);
        }

        self.dispatch_compute_pipelines(&[&self.update_positions]);
        //println!("{:?}", format_particles!(self.buffer, position));
        if env::var("XCODE").is_ok() {
            metal_context.capture_scope.end_scope();

        }
        self.run = false;
    }


    fn get_neighbors(&mut self, metal_context: &MetalContext) {
        self.dispatch_compute_pipelines(&[&self.compute_cell]);
        let mut cells = unsafe { std::slice::from_raw_parts_mut(self.cell_buffer.contents() as *mut ParticleLocation, self.num_particles)};

        cells.sort_by(|a, b| a.cell.cmp(&b.cell));
        //println!("{:?}", cells);
        let mut cell_indices = unsafe { std::slice::from_raw_parts_mut(self.cell_index_buffer.contents() as *mut u32, self.cell_index_buffer.length() as usize / size_of::<u32>())};
        let mut last = u32::MAX;
        let mut i = 0;
        cell_indices.fill(u32::MAX);
        for cell in cells.iter() {
            if cell.cell != last {
                cell_indices[cell.cell as usize] = i;
                last = cell.cell;
            }
            i += 1
        }
        //println!(" cell {:?}", cell_indices);
        //println!(" cell {:?}", cell_indices[10196]);
    }

    fn correct_density(&mut self, metal_context: &MetalContext) {
        let mut iterations = 0;
        let error = 10f32;  // percent

        self.dispatch_compute_pipelines(&[&self.compute_density_adv]);
        if unsafe {(self.buffer.contents() as *const Particle).as_ref().unwrap().density_adv.is_nan()} {
            if env::var("XCODE").is_ok() {
                metal_context.capture_scope.end_scope();
                self.run = false;
            }
            //exit(0);
        }

        //println!("{:?}", format_particles!(self.buffer, factor));

        while {
            let cont = unsafe { (compute_args!(self.arg_buffer, density_error)) >= (error * 0.001f32 * compute_args!(self.arg_buffer, rest_density)) } ;
            if (iterations > 2 || compute_args!(self.arg_buffer, density_error) > 0.0001 || true) {
                println!("density_error: {}", compute_args!(self.arg_buffer, density_error));

                println!("iterations: {}", iterations);
            }

            if compute_args!(self.arg_buffer, density_error).is_infinite() {
                exit(0);
            }
            compute_args!(self.arg_buffer, density_error, 0.);
            (iterations < 2 || cont) && !(iterations > 200)
        } {
            self.dispatch_compute_pipelines(&[&self.compute_pressure_accel, &self.correct_density_error]);
            iterations += 1;
        }

        self.dispatch_compute_pipelines(&[&self.compute_pressure_accel, &self.update_velocities_from_pressure]);
    }

    fn correct_divergence(&mut self) {
        let mut iterations = 0;
        let error = 10f32;  // percent

        self.dispatch_compute_pipelines(&[&self.compute_density_change]);

        while {
            let cont = unsafe { (compute_args!(self.arg_buffer, density_error)) >= (error * 0.01f32 * compute_args!(self.arg_buffer, rest_density)) } ;
            if (iterations != 0) {
                println!("divergence_error: {}", compute_args!(self.arg_buffer, density_error));
                println!("iterations: {}", iterations);
            }
            if compute_args!(self.arg_buffer, density_error).is_infinite() {
                println!("{:?}", format_particles!(self.buffer, density));
                exit(0);
            }
            //println!("{:?}", format_particles!(self.buffer, position));
            compute_args!(self.arg_buffer, density_error, 0.);

            (iterations < 1 || cont) && !(iterations > 200)
        } {
            self.dispatch_compute_pipelines(&[&self.compute_pressure_accel2]);
            self.dispatch_compute_pipelines(&[&self.correct_divergence_error]);
            iterations += 1;
        }

        self.dispatch_compute_pipelines(&[&self.compute_pressure_accel_factor]);
    }

    pub fn update_compute_arguments(&mut self) {
        compute_args!(self.arg_buffer, time_step) = unsafe { std::mem::transmute(0.005f32) };
        compute_args!(self.arg_buffer, avg_density_derivative) = 0.;
        compute_args!(self.arg_buffer, density_error) = 0.;
    }

    fn dispatch_compute<F>(&self, func: F)
    where
        F: Fn(&CommandBufferRef, &ComputeCommandEncoderRef),
    {
        let command_buffer = self.command_queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();
        command_encoder.set_buffer(0, Some(&self.buffer), 0);
        command_encoder.set_buffer(1, Some(&self.arg_buffer), 0);
        command_encoder.set_buffer(2, Some(&self.cell_buffer), 0);
        command_encoder.set_buffer(3, Some(&self.cell_index_buffer), 0);
        func(command_buffer, command_encoder);

        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    fn dispatch_compute_pipelines(&self, pipelines: &[&ComputePipeline]) {
        self.dispatch_compute(|command_buffer, command_encoder| {
            let label = pipelines.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ");
            command_buffer.set_label(label.as_str());
            command_encoder.set_label(label.as_str());
            for pipeline in pipelines {
                pipeline.dispatch(command_encoder, self.num_particles);
            }
        })
    }
}
