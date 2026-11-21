//! 把岛仔图标作为 RT_GROUP_ICON #1 编进 exe，供 LoadImageW(hInst, 1, ...) 取用。
//! MSVC 目标用 Windows SDK 的 rc.exe；没有 SDK 时退回 MinGW 的 windres（GNU 目标带得动）。
//! 两者都找不到就跳过，只少一个图标，不影响编译。

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn find_rc() -> Option<PathBuf> {
    let roots = [
        r"C:\Program Files (x86)\Windows Kits\10\bin",
        r"C:\Program Files\Windows Kits\10\bin",
    ];
    let mut found: Vec<PathBuf> = Vec::new();
    for root in roots {
        let Ok(entries) = fs::read_dir(root) else { continue };
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let rc = dir.join("x64").join("rc.exe");
            if rc.exists() {
                found.push(rc);
            }
        }
    }
    found.sort();
    found.pop()
}

fn find_windres() -> Option<PathBuf> {
    if let Some(explicit) = env::var_os("WINDRES") {
        let path = PathBuf::from(explicit);
        if path.exists() {
            return Some(path);
        }
    }
    let path = env::var_os("PATH")?;
    env::split_paths(&path).map(|dir| dir.join("windres.exe")).find(|candidate| candidate.exists())
}

fn main() {
    println!("cargo:rerun-if-changed=app.rc");
    println!("cargo:rerun-if-changed=assets/daozai.ico");
    println!("cargo:rerun-if-changed=build.rs");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    // GNU 目标走 mingw 链接时，shcore / winsqlite3 的导入库不在 rustc 自带的那一套里。
    // 仓库内自带一份，只追加搜索路径，不动 RUSTFLAGS（否则整棵依赖树都要重编译）。
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu") {
        let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("vendor").join("gnu");
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // rc.exe 产出 .res，windres 产出 COFF 对象；两者都能直接交给链接器。
    let compiled = if let Some(rc) = find_rc() {
        let res = out.join("daozai.res");
        let ok = Command::new(&rc).arg("/nologo").arg("/fo").arg(&res).arg("app.rc").status().map(|s| s.success()).unwrap_or(false);
        (ok, res)
    } else if let Some(windres) = find_windres() {
        let object = out.join("daozai-res.o");
        let ok = Command::new(&windres)
            .arg("-i")
            .arg("app.rc")
            .arg("-o")
            .arg(&object)
            .arg("--target=pe-x86-64")
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        (ok, object)
    } else {
        println!("cargo:warning=找不到 rc.exe 或 windres，跳过图标资源");
        return;
    };
    if !compiled.0 {
        println!("cargo:warning=编译 app.rc 失败，跳过图标资源");
        return;
    }
    println!("cargo:rustc-link-arg={}", compiled.1.display());
}
