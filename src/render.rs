use std::{cell::RefCell, sync::Arc};

use crate::shaders;

use anyhow::{anyhow, bail, Context, Result};
use na::{point, vector};
use nalgebra as na;
use vk::{command_buffer::PrimaryCommandBufferAbstract, pipeline::Pipeline, sync::GpuFuture};
use vulkano as vk;
use vulkano_util::context::VulkanoContext;

/// The renderer structure. Put variables needed for rendering here.
pub struct Renderer {
    /// The Vulkan context, holding our instance, device, queue, etc.
    context: VulkanoContext,

    /// The surface to render to.
    surface: Arc<vk::swapchain::Surface>,
    /// The format of the color buffer.
    color_format: vk::format::Format,
    /// The format of the debpth buffer.
    depth_format: vk::format::Format,
    /// The swapchain to use, and the buffers bound to that swapchain.
    framebuffers: RefCell<Framebuffers>,

    /// The configuration of our render pass.
    render_pass: Arc<vk::render_pass::RenderPass>,
    /// Our graphics pipeline, specifying the settings and shaders to use
    /// to transform vertex attributes into pixels on the screen.
    pipeline: Arc<vk::pipeline::GraphicsPipeline>,
    command_buffer_allocator: vk::command_buffer::allocator::StandardCommandBufferAllocator,

    descriptor_set_allocator: vk::descriptor_set::allocator::StandardDescriptorSetAllocator,
    /// Our occasionally-updated uniforms specifying the scene settings.
    sporadic_uniforms: shaders::sample::ty::sporadicBuf,
    /// Whether we need to update our sporadic uniform buffer.
    sporadic_needs_update: bool,
    /// The data buffer storing our sporadic uniforms in GPU memory.
    sporadic_uniform_buffer: Arc<vk::buffer::DeviceLocalBuffer<shaders::sample::ty::sporadicBuf>>,
    /// The descriptor set binding our sporadic uniforms to the pipeline.
    sporadic_descriptor_set: Arc<vk::descriptor_set::PersistentDescriptorSet>,

    /// The object uniforms for our object.
    object_uniforms: shaders::sample::ty::objectBuf,

    /// The uniform buffer for our scene uniforms, stored in device memory and updated each frame.
    scene_uniform_buffer: vk::buffer::CpuBufferPool<shaders::sample::ty::sceneBuf>,
    /// The uniform buffer for our object uniforms in device memory.
    object_uniform_buffer_layout: Arc<vk::descriptor_set::layout::DescriptorSetLayout>,
    /// The layout of the descriptor set for the scene uniform buffer.
    scene_uniform_buffer_layout: Arc<vk::descriptor_set::layout::DescriptorSetLayout>,
    /// The layout of the descriptor set for the object uniform buffer.
    object_uniform_buffer: vk::buffer::CpuBufferPool<shaders::sample::ty::objectBuf>,

    /// The descriptor set binding our texture image and sampler.
    sampler_descriptor_set: Arc<vk::descriptor_set::PersistentDescriptorSet>,

    /// The projection matrix.
    projection: Option<na::Matrix4<f32>>,
    /// The view matrix.
    view: na::Matrix4<f32>,
    /// The scene-orientation matrix.
    scene_orient: na::Matrix4<f32>,

    /// The rotation the user input via the mouse.
    mouse_rotation: na::Vector2<f32>,
    // The scale the user input via the mouse.
    mouse_scale: f32,

    /// The variables controlling animation playback.
    animation_state: AnimationState,
    /// Whether the user has enabled automatic rotation.
    use_rotation: bool,

    /// Our vertex buffer, stored in device memory.
    vertex_buffer: Arc<vk::buffer::DeviceLocalBuffer<[shaders::sample::Attributes]>>,
    /// The number of vertices in the vertex buffer.
    num_vertices: u32,

    /// The future representing the time at which the previous frame will finish rendering.
    ///
    /// Since GPU work happens asyncronously, Vulkano uses "futures" to represent the status of
    /// this work. Whenever we submit a commmand buffer we're given a "future" representing the
    /// moment in time that the command buffer finishes executing -- as the name implies, this time
    /// may be indefinitely in the future. Later on, we can synchronize events (such as the
    /// rendering of the next frame) to occur after this future completes.
    previous_frame: Option<Box<dyn vk::sync::GpuFuture>>,

    previous_frame_fence: Option<Arc<vk::sync::Fence>>,
}

/// The status of our animation.
enum AnimationState {
    /// The animation is running. `start` is the moment at which t=0.
    Running { start: std::time::Instant },

    /// The animation is paused, at t=`elapsed`.
    Paused { elapsed: std::time::Duration },
}

impl Renderer {
    /// Called before rendering each frame to update animated variables.
    pub fn animate(&mut self) -> f32 {
        let duration = match self.animation_state {
            AnimationState::Running { start } => start.elapsed(),
            AnimationState::Paused { elapsed } => elapsed,
        };
        let time = duration.as_secs_f32();

        const SECONDS_PER_CYCLE: f32 = 3.;

        // Scale the scene by our scale factor.
        self.scene_orient =
            na::Scale3::new(self.mouse_scale, self.mouse_scale, self.mouse_scale).into();

        // Rotate the scene around by either the mouse rotation or the animated rotation.
        if self.use_rotation {
            let gradual_rotation: na::Matrix4<f32> = na::Rotation::from_axis_angle(
                &na::UnitVector3::new_unchecked(na::Vector3::y()),
                2. * std::f32::consts::PI * time / SECONDS_PER_CYCLE,
            )
            .into();
            self.scene_orient *= gradual_rotation;
        } else {
            let mouse_rotation_x: na::Matrix4<f32> = na::Rotation::from_axis_angle(
                &na::UnitVector3::new_unchecked(na::Vector3::x()),
                self.mouse_rotation.x,
            )
            .into();
            let mouse_rotation_y: na::Matrix4<f32> = na::Rotation::from_axis_angle(
                &na::UnitVector3::new_unchecked(na::Vector3::y()),
                self.mouse_rotation.y,
            )
            .into();

            self.scene_orient *= mouse_rotation_x * mouse_rotation_y;
        }

        // Return the animation time, since that goes in the scene uniform buffer.
        time
    }

    pub fn render(&mut self) -> Result<()> {
        // Cycle the animation.
        let time = self.animate();

        // Create a command buffer to store our rendering commands.
        let mut command_buffer = vk::command_buffer::AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.context.graphics_queue().queue_family_index(),
            vk::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        // Acquire a framebuffer we can render into.. This will (re-)create the framebuffers and
        // swapchain if necessary. If all framebuffers are currently in use by the GPU, this will
        // block until one becomes available.
        let framebuffer = self.framebuffers.borrow_mut().next(self)?;

        if self.sporadic_needs_update {
            // Our sporadic uniforms have changed; write the new values to the uniform buffer.
            command_buffer.update_buffer(
                Box::new(self.sporadic_uniforms),
                self.sporadic_uniform_buffer.clone(),
                0,
            )?;
            self.sporadic_needs_update = false;
        }

        // Start the render pass by binding our framebuffer
        // and clearing the color and depth attachments.
        command_buffer.begin_render_pass(
            vk::command_buffer::RenderPassBeginInfo {
                clear_values: vec![
                    Some(vk::format::ClearValue::Float([0., 0., 0., 0.])),
                    Some(vk::format::ClearValue::Depth(1.)),
                ],
                ..vk::command_buffer::RenderPassBeginInfo::framebuffer(framebuffer.framebuffer)
            },
            vk::command_buffer::SubpassContents::Inline,
        )?;

        // Configure our viewport dimensions.
        let dimensions = framebuffer.swapchain.image_extent();
        let dimensions = [dimensions[0] as f32, dimensions[1] as f32];
        command_buffer.set_viewport(
            0,
            [vk::pipeline::graphics::viewport::Viewport {
                origin: [0.0, 0.0],
                dimensions,
                depth_range: 0.0..1.0,
            }],
        );

        // Use our graphics pipeline.
        command_buffer.bind_pipeline_graphics(self.pipeline.clone());

        // If we do not already have a projection matrix, create one.
        let projection = self.projection.get_or_insert_with(|| {
            let mut projection = na::Matrix4::new_perspective(
                dimensions[0] / dimensions[1],
                std::f32::consts::FRAC_PI_2,
                0.1,
                100.,
            );
            projection.m22 *= -1.;
            projection
        });

        // Define our uniform buffers data.
        let scene_data = shaders::sample::ty::sceneBuf {
            uProjection: (*projection).into(),
            uView: self.view.into(),
            uSceneOrient: self.scene_orient.into(),
            uLightPos: na::Point4::new(-50., 50., 10., 1.).into(),
            uLightColor: na::Vector4::new(1., 1., 1., 1.).into(),
            uLightKaKdKs: na::Vector4::new(0.2, 0.5, 0.3, 1.).into(),
            uTime: time,
        };

        // Upload our uniform buffers to the GPU, and create descriptor sets.
        let scene_descriptor_set = vk::descriptor_set::PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.scene_uniform_buffer_layout.clone(),
            [vk::descriptor_set::WriteDescriptorSet::buffer(
                0,
                // Our uniform buffers are `CpuBufferPool`s, which is a helper type included in
                // Vulkano that manages a pool of host-visible device-local buffers. It
                // automatically handles double-buffering, so we can write to one buffer in the
                // pool while the GPU is still rendering the previous frame using another buffer.
                self.scene_uniform_buffer.from_data(scene_data)?,
            )],
        )?;
        let object_descriptor_set = vk::descriptor_set::PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.object_uniform_buffer_layout.clone(),
            [vk::descriptor_set::WriteDescriptorSet::buffer(
                0,
                self.object_uniform_buffer.from_data(self.object_uniforms)?,
            )],
        )?;

        // Bind our descriptor sets to the pipeline.
        command_buffer.bind_descriptor_sets(
            vk::pipeline::PipelineBindPoint::Graphics,
            self.pipeline.layout().clone(),
            0,
            (
                self.sporadic_descriptor_set.clone(),
                scene_descriptor_set,
                object_descriptor_set,
                self.sampler_descriptor_set.clone(),
            ),
        );

        // Draw our geometry.
        command_buffer.bind_vertex_buffers(0, [self.vertex_buffer.clone()]);
        command_buffer.draw(self.num_vertices, 1, 0, 0)?;

        // we're done rendering!
        command_buffer.end_render_pass()?;

        // Now, we need to upload our command buffer to the GPU for processing.
        // But we can't start processing this frame until the previous frame completes...
        let previous_frame = self
            .previous_frame
            .take()
            .unwrap_or_else(|| vk::sync::now(self.context.device().clone()).boxed());

        // ...and the next framebuffer is available.
        let ready = previous_frame.join(framebuffer.ready);

        // Submit the command buffer for execution after the above conditions are met.
        let mut inflight = command_buffer
            .build()?
            .execute_after(ready, self.context.graphics_queue().clone())?
            .then_swapchain_present(
                self.context.graphics_queue().clone(),
                vk::swapchain::SwapchainPresentInfo::swapchain_image_index(
                    framebuffer.swapchain,
                    framebuffer.index as u32,
                ),
            );

        match inflight.flush() {
            Ok(_) => {
                // Our frame is being processed!
                inflight.cleanup_finished();
                self.previous_frame = Some(inflight.boxed())
            }
            Err(vk::sync::FlushError::OutOfDate) => {
                // This frame could not be rendered because the swapchain is no longer valid.
                // Make sure we recreate it next time.
                self.framebuffers.borrow_mut().invalidate();
            }
            Err(e) => bail!(e), // Something went wrong; return the error so we can print it.
        }

        // Signal a fence after the frame is finished.
        let fence = Arc::new(vk::sync::Fence::from_pool(self.context.device().clone())?);
        unsafe {
            // This seems to be the only way to do this at the moment...
            self.context
                .graphics_queue()
                .with(|mut q| q.submit_unchecked([Default::default()], Some(fence.clone())))?;
        }

        // Wait for the previous frame to finish rendering, so we don't get a ton of them piled up.
        if let Some(fence) = self.previous_frame_fence.take() {
            fence.wait(None)?;
        }
        self.previous_frame_fence = Some(fence);

        Ok(())
    }

    /// Called when the dimensions of the window change.
    pub fn resize(&mut self) {
        // We'll need to recreate the swapchain and projection matrix before we can render again.
        self.framebuffers.get_mut().invalidate();
        self.projection = None;
    }

    pub fn mouse_dragged(&mut self, dx: f32, dy: f32) {
        const ANGFACT: f32 = std::f32::consts::PI / 180.;
        self.mouse_rotation += vector![-dy, dx] * ANGFACT;
    }

    pub fn mouse_scrolled(&mut self, _dx: f32, dy: f32) {
        const SCLFACT: f32 = 0.005;
        const MINSCALE: f32 = 0.05;
        self.mouse_scale += dy * SCLFACT;
        self.mouse_scale = self.mouse_scale.max(MINSCALE);
    }

    /// Called when the user toggles lighting off or on.
    pub fn toggle_lighting(&mut self) {
        self.sporadic_uniforms.uUseLighting = (self.sporadic_uniforms.uUseLighting == 0) as i32;
        self.sporadic_needs_update = true;
    }

    /// Called when the user toggles texturing off or on.
    pub fn toggle_mode(&mut self) {
        self.sporadic_uniforms.uMode = (self.sporadic_uniforms.uMode == 0) as i32;
        self.sporadic_needs_update = true;
    }

    /// Called when the user presses the "pause" key.
    #[allow(clippy::unchecked_duration_subtraction)] // overly pedantic lint that will
                                                     // be disabled by default in Rust 1.68
    pub fn toggle_paused(&mut self) {
        self.animation_state = match self.animation_state {
            AnimationState::Running { start } => AnimationState::Paused {
                elapsed: start.elapsed(),
            },
            AnimationState::Paused { elapsed } => AnimationState::Running {
                start: std::time::Instant::now() - elapsed,
            },
        }
    }

    /// Called when the user toggles rotation off or on.
    pub fn toggle_rotation(&mut self) {
        self.use_rotation = !self.use_rotation;
    }

    /// Creates our geometry.
    fn vertices() -> Vec<shaders::sample::Attributes> {
        use shaders::sample::Attributes;
        vec![
            // front face of pyramid
            Attributes {
                aVertex: [0., 1., 0.],
                aNormal: vector![0., 1., 1.].normalize().into(),
                aColor: [1., 0., 0.],
                aTexCoord: [0.5, 0.],
            },
            Attributes {
                aVertex: [1., 0., 1.],
                aNormal: vector![0., 1., 1.].normalize().into(),
                aColor: [1., 0., 0.],
                aTexCoord: [1., 1.],
            },
            Attributes {
                aVertex: [-1., 0., 1.],
                aNormal: vector![0., 1., 1.].normalize().into(),
                aColor: [1., 0., 0.],
                aTexCoord: [0., 1.],
            },
            // left face
            Attributes {
                aVertex: [0., 1., 0.],
                aNormal: vector![-1., 1., 0.].normalize().into(),
                aColor: [0., 1., 0.],
                aTexCoord: [0.5, 0.],
            },
            Attributes {
                aVertex: [-1., 0., 1.],
                aNormal: vector![-1., 1., 0.].normalize().into(),
                aColor: [0., 1., 0.],
                aTexCoord: [1., 1.],
            },
            Attributes {
                aVertex: [-1., 0., -1.],
                aNormal: vector![-1., 1., 0.].normalize().into(),
                aColor: [0., 1., 0.],
                aTexCoord: [0., 1.],
            },
            // right face
            Attributes {
                aVertex: [0., 1., 0.],
                aNormal: vector![1., 1., 0.].normalize().into(),
                aColor: [0., 0., 1.],
                aTexCoord: [0.5, 0.],
            },
            Attributes {
                aVertex: [1., 0., -1.],
                aNormal: vector![1., 1., 0.].normalize().into(),
                aColor: [0., 0., 1.],
                aTexCoord: [0., 1.],
            },
            Attributes {
                aVertex: [1., 0., 1.],
                aNormal: vector![1., 1., 0.].normalize().into(),
                aColor: [0., 0., 1.],
                aTexCoord: [1., 1.],
            },
            // back face
            Attributes {
                aVertex: [0., 1., 0.],
                aNormal: vector![0., 1., -1.].normalize().into(),
                aColor: [1., 1., 0.],
                aTexCoord: [0.5, 0.],
            },
            Attributes {
                aVertex: [1., 0., -1.],
                aNormal: vector![0., 1., -1.].normalize().into(),
                aColor: [1., 1., 0.],
                aTexCoord: [0., 1.],
            },
            Attributes {
                aVertex: [-1., 0., -1.],
                aNormal: vector![0., 1., -1.].normalize().into(),
                aColor: [1., 1., 0.],
                aTexCoord: [1., 1.],
            },
            // bottom
            Attributes {
                aVertex: [1., 0., 1.],
                aNormal: [0., -1., 0.],
                aColor: [0., 1., 1.],
                aTexCoord: [1., 0.],
            },
            Attributes {
                aVertex: [1., 0., -1.],
                aNormal: [0., -1., 0.],
                aColor: [0., 1., 1.],
                aTexCoord: [1., 1.],
            },
            Attributes {
                aVertex: [-1., 0., -1.],
                aNormal: [0., -1., 0.],
                aColor: [0., 1., 1.],
                aTexCoord: [0., 0.],
            },
            Attributes {
                aVertex: [-1., 0., -1.],
                aNormal: [0., -1., 0.],
                aColor: [0., 1., 1.],
                aTexCoord: [0., 0.],
            },
            Attributes {
                aVertex: [-1., 0., 1.],
                aNormal: [0., -1., 0.],
                aColor: [0., 1., 1.],
                aTexCoord: [1., 0.],
            },
            Attributes {
                aVertex: [1., 0., 1.],
                aNormal: [0., -1., 0.],
                aColor: [0., 1., 1.],
                aTexCoord: [1., 0.],
            },
        ]
    }

    /// Initializes all rendering resources.
    pub fn new(context: VulkanoContext, surface: Arc<vk::swapchain::Surface>) -> Result<Self> {
        // Choose a color and depth format.
        let color_format = context
            .device()
            .physical_device()
            .surface_formats(&surface, Default::default())?
            .iter()
            .find(|(f, c)| {
                *c == vk::swapchain::ColorSpace::SrgbNonLinear
                    && [
                        vk::format::Format::R8G8B8A8_SRGB,
                        vk::format::Format::B8G8R8A8_SRGB,
                    ]
                    .contains(f)
            })
            .ok_or_else(|| anyhow!("no suitable color formats"))?
            .0;
        let depth_format = vk::format::Format::D16_UNORM;

        // Define our render pass.
        let render_pass = vulkano::single_pass_renderpass!(context.device().clone(),
            attachments: {
                // Clear the color buffer on load, and store it to memory so we can see it.
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                },
                // Clear the depth buffer on load, but we don't need to save the results.
                // This is a significant performance optimization on GPUs that support tiled
                // rendering (i.e. completely rendering a small tile of the screen before moving
                // onto the next tile), because it means the depth buffer can be stored in registers
                // on the GPU and never actually written to off-chip memory.
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: depth_format,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )?;

        // Define our graphics pipeline:
        let pipeline = vk::pipeline::GraphicsPipeline::start()
            // Use the render pass we described above
            .render_pass(
                vk::pipeline::graphics::render_pass::PipelineRenderPassType::BeginRenderPass(
                    render_pass.clone().first_subpass(),
                ),
            )
            // The vertex inputs are defined as in our shader attributes.
            .vertex_input_state(
                vk::pipeline::graphics::vertex_input::BuffersDefinition::new()
                    .vertex::<shaders::sample::Attributes>(),
            )
            // Pass the inputs through our vertex shader
            .vertex_shader(
                shaders::sample::load_vertex(context.device().clone())?
                    .entry_point("main")
                    .context("entry point not found")?,
                (),
            )
            // Pass the pixels through our fragment shader
            .fragment_shader(
                shaders::sample::load_fragment(context.device().clone())?
                    .entry_point("main")
                    .context("entry point not found")?,
                (),
            )
            // Use a basic depth test, and no stencil buffer.
            .depth_stencil_state(vk::pipeline::graphics::depth_stencil::DepthStencilState::simple_depth_test())
            // We'll define the viewport when we render, and we won't use a scissr test.
            .viewport_state(
                vk::pipeline::graphics::viewport::ViewportState::viewport_dynamic_scissor_irrelevant(),
            )
            .build(context.device().clone())?;

        // Now let's set up our descriptor sets.
        let descriptor_set_allocator =
            vk::descriptor_set::allocator::StandardDescriptorSetAllocator::new(
                context.device().clone(),
            );

        // Define the descriptor set layout for our uniform buffers.
        let sporadic_uniform_buffer_layout = vk::descriptor_set::layout::DescriptorSetLayout::new(
            context.device().clone(),
            vk::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    vk::descriptor_set::layout::DescriptorSetLayoutBinding {
                        stages: vk::shader::ShaderStage::Fragment.into(),
                        ..vk::descriptor_set::layout::DescriptorSetLayoutBinding::descriptor_type(
                            vk::descriptor_set::layout::DescriptorType::UniformBuffer,
                        )
                    },
                )]
                .into(),
                ..Default::default()
            },
        )?;
        let scene_uniform_buffer_layout = vk::descriptor_set::layout::DescriptorSetLayout::new(
            context.device().clone(),
            vk::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    vk::descriptor_set::layout::DescriptorSetLayoutBinding {
                        stages: vk::shader::ShaderStages {
                            vertex: true,
                            fragment: true,
                            ..Default::default()
                        },
                        ..vk::descriptor_set::layout::DescriptorSetLayoutBinding::descriptor_type(
                            vk::descriptor_set::layout::DescriptorType::UniformBuffer,
                        )
                    },
                )]
                .into(),
                ..Default::default()
            },
        )?;
        let object_uniform_buffer_layout = vk::descriptor_set::layout::DescriptorSetLayout::new(
            context.device().clone(),
            vk::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    vk::descriptor_set::layout::DescriptorSetLayoutBinding {
                        stages: vk::shader::ShaderStages {
                            vertex: true,
                            fragment: true,
                            ..Default::default()
                        },
                        ..vk::descriptor_set::layout::DescriptorSetLayoutBinding::descriptor_type(
                            vk::descriptor_set::layout::DescriptorType::UniformBuffer,
                        )
                    },
                )]
                .into(),
                ..Default::default()
            },
        )?;

        let object_uniforms = shaders::sample::ty::objectBuf {
            uModel: na::Matrix4::identity().into(),
            uNormal: na::Matrix4::identity().into(),
            uColor: [1., 0., 0., 1.],
            uShininess: 10.,
        };

        // Allocate the sporadic uniform buffer.
        // Since we rarely need to write to it, we can make it device-local
        // and issue upload commands when we need to access it.
        let sporadic_uniform_buffer = vk::buffer::DeviceLocalBuffer::new(
            context.memory_allocator(),
            vk::buffer::BufferUsage {
                uniform_buffer: true,
                transfer_dst: true,
                ..Default::default()
            },
            context
                .device()
                .active_queue_family_indices()
                .iter()
                .copied(),
        )?;

        // Create a descriptor set for the sporadic uniform buffer.
        let sporadic_descriptor_set = vk::descriptor_set::PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            sporadic_uniform_buffer_layout,
            [vk::descriptor_set::WriteDescriptorSet::buffer(
                0,
                sporadic_uniform_buffer.clone(),
            )],
        )?;

        // Allocate the scene and object uniform buffers.
        // Since they will be updated every frame, we'll create a pool of buffers that live in
        // device memory but can be written to by the CPU. Using a pool instead of a single buffer
        // allows us to write to the buffers for the next frame while the GPU is still rendering
        // the current frame.
        let scene_uniform_buffer =
            vk::buffer::CpuBufferPool::uniform_buffer(context.memory_allocator().clone());
        let object_uniform_buffer =
            vk::buffer::CpuBufferPool::uniform_buffer(context.memory_allocator().clone());

        // Create a command buffer so we can upload our vertex buffer & texture.
        let command_buffer_allocator =
            vk::command_buffer::allocator::StandardCommandBufferAllocator::new(
                context.device().clone(),
                Default::default(),
            );
        let mut command_buffer = vk::command_buffer::AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            context.graphics_queue().queue_family_index(),
            vk::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )?;

        // Create the vertex buffer.
        let vertices = Renderer::vertices();
        let num_vertices = vertices.len();
        let vertex_buffer = vk::buffer::DeviceLocalBuffer::from_iter(
            context.memory_allocator(),
            vertices.into_iter(),
            vk::buffer::BufferUsage {
                vertex_buffer: true,
                ..Default::default()
            },
            &mut command_buffer,
        )?;

        // Load the texture.

        // Load the image from disk.
        let texture_image = image::open("mikebailey.jpg")?.into_rgba8();
        let (width, height) = texture_image.dimensions();

        // Create an image buffer in GPU memory, and upload it to the GPU.
        let texture_image = Arc::new(vk::image::ImmutableImage::from_iter(
            context.memory_allocator(),
            texture_image.into_raw(),
            vk::image::ImageDimensions::Dim2d {
                width,
                height,
                array_layers: 1,
            },
            vk::image::MipmapsCount::Log2,
            vk::format::Format::R8G8B8A8_UNORM,
            &mut command_buffer,
        )?);

        let texture_image = vk::image::view::ImageView::new(
            texture_image.clone(),
            vk::image::view::ImageViewCreateInfo::from_image(&texture_image),
        )?;

        // Create a sampler that describes how to access our image.
        let sampler = vk::sampler::Sampler::new(
            context.device().clone(),
            vk::sampler::SamplerCreateInfo {
                mag_filter: vk::sampler::Filter::Linear,
                min_filter: vk::sampler::Filter::Linear,
                ..Default::default()
            },
        )?;

        // Create a descriptor set binding our texture & sampler to the graphics pipeline.
        let sampler_descriptor_set = vk::descriptor_set::PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            vk::descriptor_set::layout::DescriptorSetLayout::new(
                context.device().clone(),
                vk::descriptor_set::layout::DescriptorSetLayoutCreateInfo {
                    bindings: [(
                        0,
                        vk::descriptor_set::layout::DescriptorSetLayoutBinding {
                            stages: vk::shader::ShaderStage::Fragment.into(),
                            ..vk::descriptor_set::layout::DescriptorSetLayoutBinding::descriptor_type(
                                vk::descriptor_set::layout::DescriptorType::CombinedImageSampler)
                        },
                    )]
                    .into(),
                    ..Default::default()
                },
            )?,
            [
                vk::descriptor_set::WriteDescriptorSet::image_view_sampler(0, texture_image, sampler)
            ],
        )?;

        // Submit our upload commands to the GPU.
        let previous_frame = command_buffer
            .build()?
            .execute(context.graphics_queue().clone())?
            .boxed();
        previous_frame.flush()?;

        Ok(Renderer {
            context,
            surface,

            color_format,
            depth_format,
            framebuffers: Default::default(),

            render_pass,
            pipeline,
            command_buffer_allocator,

            descriptor_set_allocator,
            sporadic_uniforms: shaders::sample::ty::sporadicBuf {
                uMode: 0,
                uUseLighting: 1,
                uNumInstances: 1,
            },
            sporadic_needs_update: true,
            object_uniforms,

            sporadic_uniform_buffer,
            sporadic_descriptor_set,
            scene_uniform_buffer,
            object_uniform_buffer_layout,
            scene_uniform_buffer_layout,
            object_uniform_buffer,
            sampler_descriptor_set,

            projection: None,
            view: na::Matrix4::look_at_rh(
                &point![0., 1., -2.],
                &na::Point3::origin(),
                &na::Vector3::y(),
            ),
            scene_orient: na::Matrix4::identity(),
            animation_state: AnimationState::Running {
                start: std::time::Instant::now(),
            },
            use_rotation: true,
            mouse_rotation: na::Vector2::zeros(),
            mouse_scale: 1.,

            vertex_buffer,
            num_vertices: num_vertices.try_into().expect("that's a lot of vertices"),

            previous_frame: Some(previous_frame),
            previous_frame_fence: None,
        })
    }
}

/// The framebuffers & swapchain to use for rendering.
enum Framebuffers {
    /// We do not have valid framebuffers.
    Invalid {
        old_swapchain: Option<Arc<vk::swapchain::Swapchain>>,
    },

    /// We have a valid swapchain with valid framebuffers.
    Valid {
        swapchain: Arc<vk::swapchain::Swapchain>,
        framebuffers: Vec<Arc<vk::render_pass::Framebuffer>>,
    },
}

/// A single framebuffer from the swapchain.
struct Framebuffer {
    /// The swapchain this framebuffer belongs to.
    pub swapchain: Arc<vk::swapchain::Swapchain>,

    /// The framebuffer itself.
    pub framebuffer: Arc<vk::render_pass::Framebuffer>,

    /// The index of the framebuffer within the swapchain.
    pub index: usize,

    /// The time at which the framebuffer will be available for us to render into.
    pub ready: vk::swapchain::SwapchainAcquireFuture,
}

impl Default for Framebuffers {
    fn default() -> Self {
        Self::Invalid {
            old_swapchain: None,
        }
    }
}

impl Framebuffers {
    /// Marks this swapchain as invalid so it will be recreated before rendering the next frame.
    pub fn invalidate(&mut self) {
        *self = match std::mem::take(self) {
            Framebuffers::Valid { swapchain, .. } => Framebuffers::Invalid {
                old_swapchain: Some(swapchain),
            },
            invalid => invalid,
        }
    }

    /// Acquires the next framebuffer in the swapchain.
    /// Recreates the swapchain if necessary. If all framebuffers are currently in use by the GPU,
    /// blocks until one becomes available.
    fn next(&mut self, renderer: &Renderer) -> Result<Framebuffer> {
        loop {
            // Do we have a valid swapchain already?
            let (swapchain, framebuffers) = match self {
                Self::Valid {
                    swapchain,
                    framebuffers,
                } => (swapchain.clone(), &*framebuffers), // Yes, use it.
                Self::Invalid { .. } => {
                    self.create(renderer)?; // No, create one.
                    continue;
                }
            };

            // Get the next framebuffer from the swapchain.
            match vk::swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok((index, suboptimal, ready)) => {
                    let index = index as usize;
                    let framebuffer = framebuffers[index].clone();
                    if suboptimal {
                        // Vulkan told us that the surface has changed so that our swapchain is no
                        // longer a perfect match. We can still render with it, but we should
                        // recreate it next time for better results.
                        self.invalidate();
                    }
                    return Ok(Framebuffer {
                        swapchain,
                        framebuffer,
                        index,
                        ready,
                    });
                }
                Err(vk::swapchain::AcquireError::OutOfDate) => {
                    // We could not acquire a framebuffer because the surface has changed such that
                    // our swapchain is invalid. Recreate the swapchain and try again.
                    self.invalidate();
                    continue;
                }
                Err(e) => bail!(e),
            }
        }
    }

    /// Creates or recreates the framebuffers and swapchain.
    pub fn create(&mut self, renderer: &Renderer) -> Result<()> {
        // If we have an old swapchain, we can reuse its resources.
        let old_swapchain = match std::mem::take(self) {
            Self::Valid { swapchain, .. } => Some(swapchain),
            Self::Invalid { old_swapchain } => old_swapchain,
        };

        // Create a chain of color buffers, and one shared depth buffer (since we need to present
        // one color buffer while rendering another, but we don't need to present the depth buffer).
        let min_image_count = renderer
            .context
            .device()
            .physical_device()
            .surface_capabilities(&renderer.surface, Default::default())?
            .min_image_count;

        let swapchain_info = vk::swapchain::SwapchainCreateInfo {
            min_image_count,
            image_format: Some(renderer.color_format),
            image_color_space: vk::swapchain::ColorSpace::SrgbNonLinear,
            image_usage: vk::image::ImageUsage {
                color_attachment: true,
                ..Default::default()
            },
            ..Default::default()
        };

        // Create a swapchain & color buffers for our surface.
        let (swapchain, color_buffers) = if let Some(old) = old_swapchain {
            old.recreate(swapchain_info)?
        } else {
            vk::swapchain::Swapchain::new(
                renderer.context.device().clone(),
                renderer.surface.clone(),
                swapchain_info,
            )?
        };

        // The depth buffer can be transient
        // (i.e. it doesn't need to be kept in memory between render passes.)
        let depth_buffer =
            vk::image::view::ImageView::new_default(vk::image::AttachmentImage::transient(
                renderer.context.memory_allocator(),
                swapchain.image_extent(),
                renderer.depth_format,
            )?)?;

        // Bind the depth buffer to the color buffers to create framebuffers.
        let framebuffers: Vec<_> = color_buffers
            .into_iter()
            .map(|color_buffer| {
                Ok(vk::render_pass::Framebuffer::new(
                    renderer.render_pass.clone(),
                    vk::render_pass::FramebufferCreateInfo {
                        attachments: vec![
                            vk::image::view::ImageView::new_default(color_buffer)?,
                            depth_buffer.clone(),
                        ],
                        ..Default::default()
                    },
                )?)
            })
            .collect::<Result<Vec<Arc<vk::render_pass::Framebuffer>>>>()?;

        *self = Framebuffers::Valid {
            swapchain,
            framebuffers,
        };

        Ok(())
    }
}
