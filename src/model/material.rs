use crate::texture::texture;

pub struct Material {
    #[allow(unused)]
    pub name: String,
    #[allow(unused)]
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}