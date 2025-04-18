use crate::MetalContext;
use glam::{Mat4, Vec3};
use metal::*;
use std::f32::consts::PI;

pub struct Camera {
    pub position: Vec3,
    forward: Vec3,
    perspective: Mat4,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.5, 0.5, -1.0),
            forward: Vec3::new(0., 0., -1.0),
            perspective: Mat4::perspective_lh(PI / 4.0, 1.0, 0.1, 100.0),
        }
    }

    pub fn get_view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward, Vec3::Y)
    }
}

pub struct RenderParticles {
    pub camera: Camera,
    num_particles: usize,
    command_queue: CommandQueue,
    cube_vertices: Buffer,
    cube_indices: Buffer,
    render_pipeline_state: RenderPipelineState,
}

impl RenderParticles {
    pub fn new(num_particles: usize, metal_context: &MetalContext) -> Self {
        let radius = 0.01f32 / 2.0;
        let cube_vertices = [
            [-radius, -radius, radius, 0.0],
            [radius, -radius, radius, 0.0],
            [-radius, radius, radius, 0.0],
            [radius, radius, radius, 0.0],
            [-radius, -radius, -radius, 0.0],
            [radius, -radius, -radius, 0.0],
            [-radius, radius, -radius, 0.0],
            [radius, radius, -radius, 0.0],
        ];

        let cube_indices = [
            // thanks https://stackoverflow.com/a/58775844
            [2u16, 6, 7],
            [2, 3, 7],
            [0, 4, 5],
            [0, 1, 5],
            [0, 2, 6],
            [0, 4, 6],
            [1, 3, 7],
            [1, 5, 7],
            [0, 2, 3],
            [0, 1, 3],
            [4, 6, 7],
            [4, 5, 7],
        ];

        let render_pipeline_desc = RenderPipelineDescriptor::new();

        render_pipeline_desc.set_vertex_function(Some(
            &metal_context
                .library
                .get_function("vertexShader", None)
                .unwrap(),
        ));
        render_pipeline_desc.set_fragment_function(Some(
            &metal_context
                .library
                .get_function("fragmentShader", None)
                .unwrap(),
        ));
        render_pipeline_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // let library = metal_context.device.new_library_with_file();

        // render_pipeline_desc.set_vertex_function()
        // metal_context.device.

        Self {
            camera: Camera::new(),
            num_particles,
            command_queue: metal_context.device.new_command_queue(),
            cube_vertices: metal_context.device.new_buffer_with_data(
                cube_vertices.as_ptr().cast(),
                size_of_val(&cube_vertices) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            cube_indices: metal_context.device.new_buffer_with_data(
                cube_indices.as_ptr().cast(),
                size_of_val(&cube_indices) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            render_pipeline_state: metal_context
                .device
                .new_render_pipeline_state(&render_pipeline_desc)
                .unwrap(),
        }
    }

    pub fn render(
        &mut self,
        buffer: &Buffer,
        drawable: &MetalDrawableRef,
        metal_context: &MetalContext,
    ) {
        let command_buffer = self.command_queue.new_command_buffer();

        let render_pass_desc = RenderPassDescriptor::new();
        render_pass_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_load_action(MTLLoadAction::Clear);
        render_pass_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_clear_color(MTLClearColor::new(
                41.0 / 255.0,
                42.0 / 255.0,
                48.0 / 255.0,
                1.0,
            ));
        render_pass_desc
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_texture(Some(drawable.texture()));

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_desc);
        command_encoder.set_render_pipeline_state(&self.render_pipeline_state);

        command_encoder.set_vertex_buffer(0, Some(&self.cube_vertices), 0);
        command_encoder.set_vertex_buffer(1, Some(buffer), 0);
        let matrix = self.camera.perspective * self.camera.get_view_matrix();
        command_encoder.set_vertex_bytes(2, size_of::<Mat4>() as NSUInteger, unsafe {
            matrix.as_ref().as_ptr().cast()
        });

        command_encoder.set_fragment_buffer(0, Some(buffer), 0);

        command_encoder.draw_indexed_primitives_instanced(
            MTLPrimitiveType::Triangle,
            36,
            MTLIndexType::UInt16,
            &self.cube_indices,
            0,
            self.num_particles as NSUInteger,
        );

        command_encoder.end_encoding();
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}
