//! 渲染方式：默认 OpenGL；部分显卡的 OpenGL 驱动在长时间运行后会崩溃或画面错乱，
//! 这时改用 Direct3D 12（兼容渲染）。崩溃处理只留一个标记文件，下次启动时再写入设置，
//! 避免在崩溃现场改写设置文件。
use std::sync::atomic::{AtomicBool, Ordering};

static COMPAT_ACTIVE: AtomicBool = AtomicBool::new(false);

fn marker() -> std::path::PathBuf { crate::model::config_dir().join("diagnostics").join("renderer-fallback.txt") }

/// 崩溃模块是不是显卡的 OpenGL 驱动（Intel igxelpicd64 / ig9icd64、NVIDIA nvoglv64、AMD atioglxx 等，
/// 以及系统的 opengl32）。只看文件名，不含路径。
pub fn is_gl_driver(module: &str) -> bool {
    let name = module.rsplit(['\\', '/']).next().unwrap_or(module).to_ascii_lowercase();
    let stem = name.strip_suffix(".dll").unwrap_or(&name);
    stem == "opengl32" || stem.contains("ogl") || (stem.contains("icd") && !stem.contains("vk"))
}

/// 本次运行实际使用的是不是兼容渲染（崩溃处理据此判断是否还需要切换）。
pub fn compat_active() -> bool { COMPAT_ACTIVE.load(Ordering::Relaxed) }
pub fn set_compat_active(active: bool) { COMPAT_ACTIVE.store(active, Ordering::Relaxed) }

/// 崩溃现场调用：只写一个小标记文件，记下崩溃的驱动模块。
pub fn request_fallback(module: &str) {
    let path = marker();
    if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
    let _ = std::fs::write(path, module.chars().filter(|c| !c.is_control()).take(120).collect::<String>());
}

/// OpenGL 驱动指纹：模块名 + 文件大小 + 修改时间。驱动换过一版，指纹就变；
/// 文件找不到时指纹恒为 `名字|0|0`（不会误判成「驱动更新过」）。
pub fn driver_stamp(module: &str) -> String {
    let name = module.rsplit(['\\', '/']).next().unwrap_or(module).to_ascii_lowercase();
    let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
    let meta = std::fs::metadata(std::path::Path::new(&root).join("System32").join(&name)).ok();
    let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let time = meta.as_ref().and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs()).unwrap_or(0);
    format!("{name}|{size}|{time}")
}

fn stamp_module(stamp: &str) -> &str { stamp.split('|').next().unwrap_or(stamp) }

/// 启动时调用：兼容渲染是上次自动切过来的、而当年崩溃的 OpenGL 驱动已经换过一版时，再给 OpenGL 一次机会。
/// 自动切换不该变成永久的额外占用（这台机器上 Direct3D 12 比 OpenGL 多约 110MB 私有内存、18 个线程）。
/// 只会重放 3 次，手动打开的兼容渲染一律不动。
pub fn retry_after_driver_update() -> Option<String> {
    let mut settings = crate::model::Settings::load();
    if !settings.render_compat || !settings.render_compat_auto || settings.render_compat_retries >= 3 { return None; }
    let recorded = settings.render_compat_gl_stamp.clone();
    if recorded.is_empty() { return None; }
    let current = driver_stamp(stamp_module(&recorded));
    if current == recorded { return None; }
    settings.render_compat = false;
    settings.render_compat_auto = false;
    settings.render_compat_gl_stamp.clear();
    settings.render_compat_reason.clear();
    settings.render_compat_retries += 1;
    settings.save();
    Some(current)
}

/// 启动时调用：有标记就打开兼容渲染并记下原因，返回原因（写入诊断用）。
pub fn apply_pending_fallback() -> Option<String> {
    let path = marker();
    let module = std::fs::read_to_string(&path).ok()?;
    let _ = std::fs::remove_file(&path);
    let module = module.trim();
    let mut settings = crate::model::Settings::load();
    if settings.render_compat { return None; }
    let when = chrono::Local::now().format("%m-%d %H:%M");
    settings.render_compat = true;
    settings.render_compat_auto = true;
    settings.render_compat_gl_stamp = driver_stamp(module);
    settings.render_compat_reason = format!("{when} 显卡的 OpenGL 驱动（{module}）崩溃，已自动切换");
    settings.save();
    Some(settings.render_compat_reason)
}

/// 本次启动是否使用兼容渲染。
pub fn compat_requested() -> bool { crate::model::Settings::load().render_compat }

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试用的配置目录放在 target 下：这台机器上 `%TEMP%` 对测试进程是拒绝访问的。
    fn test_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("test-data").join(format!("{tag}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }
    #[test]
    fn opengl_drivers_are_recognized_and_others_are_not() {
        for module in ["igxelpicd64.dll", r"C:\Windows\System32\DriverStore\FileRepository\iigd.inf\ig9icd64.dll", "ig75icd64.dll", "nvoglv64.dll", "atioglxx.dll", "OPENGL32.dll"] {
            assert!(is_gl_driver(module), "{module}");
        }
        for module in ["daozai.exe", "ntdll.dll", "KERNELBASE.dll", "d3d12.dll", "igdumdim64.dll", "nvwgf2umx.dll", "amdxc64.dll", "igvk64.dll", "vulkan-1.dll", "ucrtbase.dll"] {
            assert!(!is_gl_driver(module), "{module}");
        }
    }
    #[test]
    fn fallback_marker_turns_on_compat_rendering_once() {
        let _g = crate::ui_tests::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_dir("renderer");
        std::env::set_var("DAOZAI_DATA_DIR", &dir);
        assert!(apply_pending_fallback().is_none(), "没有标记时不改动设置");
        request_fallback("igxelpicd64.dll");
        let reason = apply_pending_fallback().expect("有标记时打开兼容渲染");
        assert!(reason.contains("igxelpicd64.dll"));
        let settings = crate::model::Settings::load();
        assert!(settings.render_compat && settings.render_compat_reason == reason);
        assert!(settings.render_compat_auto, "自动切换要记来源，驱动更新后才有机会回退");
        assert!(settings.render_compat_gl_stamp.starts_with("igxelpicd64.dll|"));
        assert!(apply_pending_fallback().is_none() && !marker().exists(), "标记只生效一次");
        std::env::remove_var("DAOZAI_DATA_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn automatic_fallback_retries_opengl_after_the_driver_changes() {
        let _g = crate::ui_tests::ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = test_dir("renderer-retry");
        std::env::set_var("DAOZAI_DATA_DIR", &dir);
        // 没有文件时指纹是 `名字|0|0`，所以 `名字|1|2` 一定和当前不同，等于「驱动换过一版」。
        let mut settings = crate::model::Settings::default();
        settings.render_compat = true;
        settings.render_compat_auto = true;
        settings.render_compat_reason = "03-01 12:00 显卡的 OpenGL 驱动（nvoglv64.dll）崩溃，已自动切换".into();
        settings.render_compat_gl_stamp = "nvoglv64.dll|1|2".into();
        settings.save();
        assert!(retry_after_driver_update().is_some(), "驱动变了就再试一次 OpenGL");
        let settings = crate::model::Settings::load();
        assert!(!settings.render_compat && !settings.render_compat_auto && settings.render_compat_gl_stamp.is_empty());
        assert_eq!(settings.render_compat_retries, 1);
        assert!(retry_after_driver_update().is_none(), "已经回到 OpenGL，不再重放");
        // 手动打开的兼容渲染不动：那是用户自己踩过坑才开的。
        let mut settings = crate::model::Settings::default();
        settings.render_compat = true;
        settings.save();
        assert!(retry_after_driver_update().is_none());
        std::env::remove_var("DAOZAI_DATA_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }
}
