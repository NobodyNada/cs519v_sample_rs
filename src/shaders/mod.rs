use bytemuck::{Pod, Zeroable};
use vulkano_shaders::shader;

pub mod sample {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    #[allow(non_snake_case)]
    pub struct Attributes {
        pub aVertex: [f32; 3],
        pub aNormal: [f32; 3],
        pub aColor: [f32; 3],
        pub aTexCoord: [f32; 2],
    }
    vulkano::impl_vertex!(Attributes, aVertex, aNormal, aColor, aTexCoord);

    shader! {
        shaders: {
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
        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
        }
    }
}
