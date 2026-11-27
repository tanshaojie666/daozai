use super::{Painter, WgpuConfiguration};
use crate::{WgpuError, WgpuSetup};
use egui::{Color32, Context, Event, RawInput, Rect, UserData, ViewportId};
use std::sync::Arc;
use winit::{
    dpi::PhysicalSize, event_loop::EventLoop, platform::windows::EventLoopBuilderExtWindows,
    window::Window,
};

/// 使用真实 DX12 队列和交换链，注入可重复的取帧失败。
pub fn verify_dx12_surface_recovery() {
    let mut config = WgpuConfiguration::default();
    if let WgpuSetup::CreateNew(setup) = &mut config.wgpu_setup {
        setup.instance_descriptor.backends = wgpu::Backends::DX12;
    }
    let ctx = Context::default();
    let mut painter = pollster::block_on(Painter::new(ctx.clone(), config, 1, None, false, false));
    if painter
        .instance
        .enumerate_adapters(wgpu::Backends::DX12)
        .is_empty()
    {
        eprintln!("当前环境没有 DX12 适配器，跳过硬件回归");
        return;
    }

    let event_loop = EventLoop::builder().with_any_thread(true).build().unwrap();
    #[allow(deprecated)]
    let window = event_loop
        .create_window(
            Window::default_attributes()
                .with_visible(false)
                .with_inner_size(PhysicalSize::new(64, 64)),
        )
        .unwrap();
    match pollster::block_on(painter.set_window(ViewportId::ROOT, Some(Arc::new(window)))) {
        Ok(()) => {}
        Err(WgpuError::NoSuitableAdapterFound(..)) => {
            eprintln!("当前环境的 DX12 适配器不支持窗口交换链，跳过硬件回归");
            return;
        }
        Err(error) => panic!("初始化 DX12 测试窗口失败：{error}"),
    }
    let state = painter.render_state().unwrap();
    assert_eq!(state.adapter.get_info().backend, wgpu::Backend::Dx12);

    let screen = Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(64., 64.));
    let mut texture = ctx.load_texture(
        "取帧失败期间更新的纹理",
        egui::ColorImage::new([1, 1], Color32::RED),
        Default::default(),
    );
    let draw = |painter: &mut Painter, image, screenshot| {
        let output = ctx.run(
            RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| {
                ctx.layer_painter(egui::LayerId::background()).image(
                    image,
                    screen,
                    Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                    Color32::WHITE,
                );
            },
        );
        let meshes = ctx.tessellate(output.shapes, output.pixels_per_point);
        assert!(!meshes.is_empty());
        painter.paint_and_update_textures(
            ViewportId::ROOT,
            output.pixels_per_point,
            [0.; 4],
            &meshes,
            &output.textures_delta,
            if screenshot {
                vec![UserData::default()]
            } else {
                Vec::new()
            },
        );
    };

    draw(&mut painter, texture.id(), false);
    state.device.poll(wgpu::Maintain::Wait);
    let buffer_count = || state.device.get_internal_counters().hal.buffers.read();
    let baseline = buffer_count();
    assert!(baseline > 0, "回归测试需要启用 DX12 缓冲计数");

    texture.set(
        egui::ColorImage::new([1, 1], Color32::GREEN),
        Default::default(),
    );
    let retired = ctx.load_texture(
        "跳帧中上传并释放的纹理",
        egui::ColorImage::new([1, 1], Color32::BLUE),
        Default::default(),
    );
    let retired_id = retired.id();
    drop(retired);
    for batch in 0..8 {
        for _ in 0..64 {
            painter.surface_error_for_test = Some(wgpu::SurfaceError::Outdated);
            draw(&mut painter, texture.id(), false);
        }
        // 已排队但从未提交的上传，即使 poll(Wait) 也不会释放。
        state.device.poll(wgpu::Maintain::Wait);
        let count = buffer_count();
        assert!(
            count <= baseline + 4,
            "第 {} 批跳帧后 DX12 缓冲持续积压：{baseline} → {count}",
            batch + 1
        );
    }
    assert!(state.renderer.read().texture(&retired_id).is_none());

    draw(&mut painter, texture.id(), true);
    state.device.poll(wgpu::Maintain::Wait);
    let mut events = Vec::new();
    painter.handle_screenshots(&mut events);
    let image = events
        .into_iter()
        .find_map(|event| match event {
            Event::Screenshot { image, .. } => Some(image),
            _ => None,
        })
        .expect("取帧恢复后应成功生成截图");
    let [width, height] = image.size;
    let pixel = image.pixels[(height / 2) * width + width / 2];
    assert_eq!(pixel, Color32::GREEN, "恢复首帧应显示跳帧期间更新的纹理");
    println!(
        "{}：连续跳过 512 帧，缓冲数 {} → {}，恢复首帧纹理正常",
        state.adapter.get_info().name,
        baseline,
        buffer_count()
    );
}
