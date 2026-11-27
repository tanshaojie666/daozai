#[cfg(windows)]
#[test]
fn dx12_skipped_frames_release_uploads_and_recover() {
    egui_wgpu::winit::verify_dx12_surface_recovery();
}
