use bytemuck::{Pod, Zeroable};
use vulkano_shaders::shader;

pub mod sample {
    use super::*;

    /// The definition of our shader attributes. These must exactly match
    /// the attributes defined in the shaders.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    #[allow(non_snake_case)]
    pub struct Attributes {
        pub aVertex: [f32; 3],
        pub aNormal: [f32; 3],
        pub aColor: [f32; 3],
        pub aTexCoord: [f32; 2],
    }

    // Vulkano can automatically generate the attribute layout from our attributes structure.
    // In a future version of Vulkano, we will be able to do this with #[derive(Vertex)] on our
    // attributes struct.
    vulkano::impl_vertex!(Attributes, aVertex, aNormal, aColor, aTexCoord);

    // Vulkano can also automatically compile our shaders for us,
    // and generate structures matching the layout of our uniforms.
    shader! {
        shaders: {
            // The shaders will share a set of constants and uniforms,
            // rather than each having their own.
            shared_constants: true,
            vertex: {
                ty: "vertex",
                path: "src/shaders/sample.vert",
            },
            fragment: {
                ty: "fragment",
                path: "src/shaders/sample.frag",
            }
        },

        // Tell Vulkano to implement some useful helper traits on our uniform structures.
        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
        }
    }
}
