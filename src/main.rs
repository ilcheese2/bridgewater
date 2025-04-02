mod render;
mod particle;
mod shader_types;

use crate::particle::SimulateParticles;
use crate::render::RenderParticles;
use cocoa::appkit::NSView;
use core_graphics_types::geometry::CGSize;
use game_loop::game_loop;
use metal::objc::runtime::YES;
use metal::{Device, Library, MTLPixelFormat, MetalLayer};
use std::sync::Arc;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::Key;
use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let window = Arc::new(event_loop.create_window(Default::default()).unwrap());

    let game = ParticleSimulation::new(window.clone());

    game_loop(event_loop, window, game, 60, 0.1, |g| {
        g.game.update();
    }, |g| {
        g.game.render(&g.window);
    }, |g, event| {
        if !g.game.your_window_handler(event) {
            g.exit();
        }
    }).expect("TODO: panic message");
}


struct ParticleSimulation {
    simulate_particles: SimulateParticles,
    render_particles: RenderParticles,
    layer: MetalLayer,
    metal_context: MetalContext
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
            library: device.new_library_with_file("/Users/cheese/RustroverProjects/bridgewater/shaders/shaders.metallib").unwrap(),
            device
        };

        Self {
            simulate_particles: SimulateParticles::new(1000, &metal_context),
            render_particles: RenderParticles::new(1000, &metal_context),
            layer,
            metal_context
        }
    }

    pub fn update(&mut self) {
       // self.simulate_particles.update();
    }

    pub fn render(&mut self, window: &Window) {
        self.render_particles.render(&self.simulate_particles.buffer, self.layer.next_drawable().unwrap(), &self.metal_context);
    }

    pub fn your_window_handler(&mut self, event: &Event<()>) -> bool {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    return false;
                },
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed { // && repeat
                        let speed = 3f32;
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
                            _ => (),
                        }
                    }
                }
                _ => {},
            },
            _ => {},
        }

        true
    }
}

struct MetalContext {
    pub device: Device,
    pub library: Library
}