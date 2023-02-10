mod render;
mod shaders;

use anyhow::Result;
use vulkano::instance;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::PhysicalPosition,
    event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WINDOW_TITLE: &str = "CS519v Rust Sample | Jonathan Keller";

fn main() -> Result<()> {
    // Create the application event loop
    let event_loop = EventLoop::new();

    // Define our Vulkan configuration
    let config = VulkanoConfig {
        instance_create_info: instance::InstanceCreateInfo {
            application_name: Some(WINDOW_TITLE.to_string()),
            enabled_extensions: instance::InstanceExtensions {
                // Enable debugging features
                ext_debug_utils: true,
                ..Default::default()
            },
            // Allow us to use devices that do not fully support
            // the Vulkan specification, such as Apple graphics hardware.
            enumerate_portability: true,
            ..Default::default()
        },

        // Use our callback to print debug messages.
        debug_create_info: Some(
            instance::debug::DebugUtilsMessengerCreateInfo::user_callback(std::sync::Arc::new(
                debug,
            )),
        ),

        // Print the name of the automatically-selected device.
        print_device_name: true,
        ..Default::default()
    };

    // Create a Vulkan instance and device with our configuration.
    let context = VulkanoContext::new(config);

    // Create a window with a Vulkan surface
    let surface = WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        .build_vk_surface(&event_loop, context.instance().clone())?;

    // Create and initialize our rendering loop.
    let mut renderer = render::Renderer::new(context, surface)?;

    // Start our event loop.
    let mut mouse_clicked = false;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                // If the user clicks "close", exit the program.
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

                // If the user resizes the window, we'll neee to recreate the framebuffers.
                WindowEvent::Resized(_) => renderer.resize(),

                // Handle keyboard input.
                WindowEvent::ReceivedCharacter(c) => match c {
                    'q' | 'Q' => *control_flow = ControlFlow::Exit,
                    'l' | 'L' => renderer.toggle_lighting(),
                    'm' | 'M' => renderer.toggle_mode(),
                    'r' | 'R' => renderer.toggle_rotation(),
                    'p' | 'P' => renderer.toggle_paused(),
                    c => println!("Received key press: {c}"),
                },

                // Handle mouse clicks.
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    mouse_clicked = state == ElementState::Pressed;
                }

                // Ignore other types of window events.
                _ => {}
            },

            Event::DeviceEvent { event, .. } => match event {
                // If the user moves the mouse while clicking, that's a drag.
                DeviceEvent::MouseMotion { delta: (dx, dy) } if mouse_clicked => {
                    renderer.mouse_dragged(dx as f32, dy as f32)
                }

                // Handle scroll events.
                DeviceEvent::MouseWheel { delta } => {
                    let (dx, dy) = match delta {
                        MouseScrollDelta::LineDelta(dx, dy) => (dx, dy),
                        MouseScrollDelta::PixelDelta(PhysicalPosition { x: dx, y: dy }) => {
                            (dx as f32, dy as f32)
                        }
                    };
                    renderer.mouse_scrolled(dx, dy);
                }

                // Ignore other types of device events.
                _ => {}
            },

            // Once we've processed all the events for this frame, render the frame.
            Event::MainEventsCleared => match renderer.render() {
                Ok(()) => {}
                Err(e) => eprintln!("[ERROR] {e:?}"),
            },

            // Ignore other types of events.
            _ => {}
        };
    })
}

/// This callback will be invoked whenever the Vulkan API generates a debug message.
fn debug(msg: &vulkano::instance::debug::Message<'_>) {
    let severity = if msg.severity.error {
        "ERROR"
    } else if msg.severity.warning {
        "WARNING"
    } else if msg.severity.information {
        "INFO"
    } else if msg.severity.verbose {
        "VERBOSE"
    } else {
        "DEBUG"
    };

    let ty = if msg.ty.validation {
        " [VALIDATION]"
    } else if msg.ty.performance {
        " [PERF]"
    } else {
        ""
    };

    eprintln!("[{severity}]{ty} {}", msg.description);
}
