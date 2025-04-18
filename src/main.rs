#![feature(repr_simd)]
mod particle;
mod render;
mod shader_types;

use crate::particle::SimulateParticles;
use crate::render::RenderParticles;
use crate::shader_types::simd_float3;
use cocoa::appkit::NSView;
use core_graphics_types::geometry::CGSize;
use game_loop::game_loop;
use imgui::Context;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use metal::objc::runtime::YES;
use metal::{CaptureManager, CaptureScope, Device, Library, MTLPixelFormat, MetalLayer};
use std::mem::offset_of;
use std::sync::Arc;
use std::time::Instant;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let window = Arc::new(event_loop.create_window(Default::default()).unwrap());

    let game = ParticleSimulation::new(window.clone());

    game_loop(
        event_loop,
        window,
        game,
        60,
        0.1,
        |g| {
            g.game.update();
        },
        |g| {
            g.game.render(&g.window);
        },
        |g, event| {
            if !g.game.window_handler(event, &g.window) {
                g.exit();
            }
        },
    )
    .expect("TODO: panic message");
}

struct ParticleSimulation {
    simulate_particles: SimulateParticles,
    render_particles: RenderParticles,
    layer: MetalLayer,
    metal_context: MetalContext,
    last_frame: std::time::Instant,
}

impl ParticleSimulation {
    pub fn new(window: Arc<Window>) -> Self {
        let device = Device::system_default().expect("device not found");

        let mut layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);

        unsafe {
            if let Ok(RawWindowHandle::AppKit(rw)) = window.window_handle().map(|wh| wh.as_raw()) {
                let view = rw.ns_view.as_ptr() as cocoa::base::id;
                view.setWantsLayer(YES);
                view.setLayer(<*mut _>::cast(layer.as_mut()));
            }
        }

        let draw_size = window.inner_size();
        layer.set_drawable_size(CGSize::new(draw_size.width as f64, draw_size.height as f64));

        let metal_context = MetalContext {
            library: device
                .new_library_with_file(
                    "shaders/shaders.metallib",
                )
                .unwrap(),
            capture_scope: CaptureManager::shared().new_capture_scope_with_device(&device),
            device,
        };
        let num_particles = 500;
        Self {
            simulate_particles: SimulateParticles::new(num_particles, &metal_context),
            render_particles: RenderParticles::new(num_particles, &metal_context),
            layer,
            metal_context,
            last_frame: Instant::now(),
        }
    }

    pub fn update(&mut self) {
        self.simulate_particles.update(&self.metal_context);
    }

    pub fn render(&mut self, window: &Window) {
        self.render_particles.render(
            &self.simulate_particles.buffer,
            self.layer.next_drawable().unwrap(),
            &self.metal_context,
        );
    }

    pub fn window_handler(&mut self, event: &Event<()>, window: &Window) -> bool {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    return false;
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed {
                        // && repeat
                        let speed = 0.2f32;
                        match event.key_without_modifiers().as_ref() {
                            Key::Character("w") => {
                                self.render_particles.camera.position.z += speed;
                            }
                            Key::Character("s") => {
                                self.render_particles.camera.position.z -= speed;
                            }
                            Key::Character("a") => {
                                self.render_particles.camera.position.x -= speed;
                            }
                            Key::Character("d") => {
                                self.render_particles.camera.position.x += speed;
                            }
                            Key::Named(NamedKey::Shift) => {
                                self.render_particles.camera.position.y -= speed;
                            }
                            Key::Named(NamedKey::Space) => {
                                self.render_particles.camera.position.y += speed;
                            }
                            _ => (),
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
        true
    }
}

struct MetalContext {
    pub device: Device,
    pub library: Library,
    pub capture_scope: CaptureScope,
}
