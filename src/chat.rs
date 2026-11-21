//! AI 对话页：多轮聊天与问答，流式输出。
//! 坐标是面板局部逻辑点（419×250）；DeepSeek 沿用「AI 额度」里已连接账号的密钥，
//! 阿里云百炼（DashScope）用自己的 API Key，存在 Windows 凭据管理器里。

use crate::app::{App, Cmd, Env, PAGE_SETTINGS_TABS, QUOTA_PAGE};
use crate::paint::{self, Card, H};
use crate::theme::{self, size, W};
use eframe::egui::{self, Pos2, Rect};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

/// 对话页的展开高度：比默认面板高，消息区和输入框各占一块。
pub(crate) const CHAT_H: f32 = 250.0;
/// 消息滚动区；滚轮停在里面时不翻页，交给区内滚动。
pub(crate) const CHAT_BODY: Rect = Rect::from_min_max(Pos2::new(42.0, 56.0), Pos2::new(377.0, 194.0));

/// DeepSeek 的 OpenAI 兼容接口；模型可在 settings.txt 里用 chat_model 覆盖。
const DEEPSEEK_ENDPOINT: &str = "https://api.deepseek.com/chat/completions";
/// 通义千问的默认基地址（阿里云百炼 DashScope）。协议是 OpenAI 兼容的那一套，
/// 用 settings.txt 里的 `aliyun_base` 可以指向任意同协议的转发站，路径部分自动补齐。
const ALIYUN_BASE: &str = "https://dashscope.aliyuncs.com";
pub(crate) const DEFAULT_MODEL: &str = "deepseek-chat";
pub(crate) const ALIYUN_MODEL: &str = "qwen-plus";
/// 默认就摆进「模型」面板的几个模型：第一次打开不用先拉接口也能直接换。
pub(crate) const DEEPSEEK_MODELS: &str = "deepseek-chat,deepseek-reasoner";
pub(crate) const ALIYUN_MODELS: &str = "qwen-plus,qwen-flash,qwen3.8-max,qwen3.7-max,qwen-max,deepseek-v4.1-flash,glm-5.3,kimi-k3,qwen3-coder-flash,qwen-long";
/// 「模型」面板最多记这么多个自己挑过的模型，防止 settings.txt 无限膨胀。
const MAX_SAVED_MODELS: usize = 40;
/// 阿里云百炼 API Key 在凭据管理器里的服务名与账号名。
const ALIYUN_SERVICE: &str = "Daozai.Aliyun";
const ALIYUN_SLOT: &str = "chat";

/// 对话服务商。两家都是 OpenAI 兼容接口，只有地址、默认模型和密钥来源不同。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Provider {
    DeepSeek,
    Aliyun,
}

impl Provider {
    pub(crate) const ALL: [Self; 2] = [Self::DeepSeek, Self::Aliyun];

    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::DeepSeek => "deepseek",
            Self::Aliyun => "aliyun",
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::DeepSeek => "DeepSeek",
            Self::Aliyun => "阿里云百炼",
        }
    }

    fn endpoint(self, base: &str) -> String {
        match self {
            Self::DeepSeek => DEEPSEEK_ENDPOINT.to_string(),
            Self::Aliyun => chat_endpoint(base),
        }
    }

    pub(crate) fn default_model(self) -> &'static str {
        match self {
            Self::DeepSeek => DEFAULT_MODEL,
            Self::Aliyun => ALIYUN_MODEL,
        }
    }

    /// 另一个服务商：点一下标题就能在两家之间切换。
    fn next(self) -> Self {
        match self {
            Self::DeepSeek => Self::Aliyun,
            Self::Aliyun => Self::DeepSeek,
        }
    }

    /// 这个服务商「自己挑过的模型」存在哪个设置项里。
    fn models_setting(self) -> &'static str {
        match self {
            Self::DeepSeek => "chat_models_deepseek",
            Self::Aliyun => "chat_models_aliyun",
        }
    }

    fn models_default(self) -> &'static str {
        match self {
            Self::DeepSeek => DEEPSEEK_MODELS,
            Self::Aliyun => ALIYUN_MODELS,
        }
    }

    pub(crate) fn parse(value: &str) -> Self {
        let value = value.trim();
        Self::ALL
            .into_iter()
            .find(|provider| provider.key().eq_ignore_ascii_case(value))
            .unwrap_or(Self::DeepSeek)
    }
}
/// 本地最多保留的消息条数，超出后从最旧的成对丢弃。
const MAX_MESSAGES: usize = 80;
/// 随请求发送的上下文字符上限，超出后只保留最近的对话。
const MAX_CONTEXT_CHARS: usize = 12000;
const SYSTEM_PROMPT: &str = "你是岛仔桌面悬浮窗里的 AI 助手，用中文回答。回答直接、具体、简短，除非用户要求展开。代码用围栏标注。不确定就直说，不要编造。";

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Role {
    User,
    Assistant,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Message {
    pub role: Role,
    pub text: String,
}

enum Event {
    Delta(String),
    Done,
    Failed(String),
}

pub(crate) struct Chat {
    pub messages: Vec<Message>,
    pub draft: String,
    pub error: String,
    pub streaming: bool,
    pub copied_at: f64,
    /// 输入框是否持有键盘焦点：翻页快捷键要避开方向键。
    pub focused: bool,
    /// 下一帧把焦点交还系统：Esc 或点岛外退出输入时用。
    pub blur: bool,
    /// 每帧请求重绘：流式输出时让新字尽快出现。
    pub scroll_bottom: bool,
    /// 正在等用户把阿里云百炼的 API Key 复制到剪贴板。
    pub key_prompt: bool,
    /// 上一次保存密钥的时间，用于短暂显示「已保存」。
    pub key_saved_at: f64,
    /// 「模型」面板：None 表示面板收起。
    pub picker: Option<Picker>,
    /// 从接口拉回来的全部模型；本次运行内缓存，关掉面板再打开不用重拉。
    pub catalog: Vec<String>,
    /// 拉取中 / 失败提示 / 是否已经试过一次。
    pub catalog_loading: bool,
    pub catalog_error: String,
    pub catalog_tried: bool,
    catalog_rx: Option<Receiver<Result<Vec<String>, String>>>,
    rx: Option<Receiver<Event>>,
    stop: Option<Arc<AtomicBool>>,
}

/// 点标题里的模型名打开的面板：只记筛选词，其它状态都在 `Chat` 上。
pub(crate) struct Picker {
    /// 筛选关键字（子串、不区分大小写）。
    pub filter: String,
}

impl Picker {
    fn new() -> Self {
        Self { filter: String::new() }
    }
}

impl Default for Chat {
    fn default() -> Self {
        Self {
            messages: load(),
            draft: String::new(),
            error: String::new(),
            streaming: false,
            copied_at: -10.0,
            focused: false,
            blur: false,
            scroll_bottom: true,
            key_prompt: false,
            key_saved_at: -10.0,
            picker: None,
            catalog: Vec::new(),
            catalog_loading: false,
            catalog_error: String::new(),
            catalog_tried: false,
            catalog_rx: None,
            rx: None,
            stop: None,
        }
    }
}

fn store() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(PathBuf::from).map(|dir| dir.join("Daozai").join("chat.json"))
}

/// 读取阿里云百炼的 API Key；没保存过返回 None。
fn aliyun_key() -> Option<String> {
    let entry = keyring::Entry::new(ALIYUN_SERVICE, ALIYUN_SLOT).ok()?;
    entry.get_password().ok().filter(|key| !key.trim().is_empty())
}

/// 保存阿里云百炼的 API Key。只进 Windows 凭据管理器，不写任何明文文件。
fn save_aliyun_key(key: &str) -> Result<(), String> {
    let key = key.trim();
    // 百炼的 Key 形如 sk-xxxxxxxx；用「足够长且不含空白」兜住误粘贴的整段文字。
    if key.len() < 20 || key.chars().any(char::is_whitespace) {
        return Err("剪贴板里不像 API Key，先在百炼控制台复制完整的密钥".into());
    }
    keyring::Entry::new(ALIYUN_SERVICE, ALIYUN_SLOT)
        .map_err(|_| "无法打开 Windows 凭据管理器".to_string())?
        .set_password(key)
        .map_err(|_| "无法保存 API Key".to_string())
}

/// 没有密钥时，告诉用户下一步该去哪。
fn missing_key(provider: Provider) -> &'static str {
    match provider {
        Provider::DeepSeek => "还没有 DeepSeek 密钥，先在「AI 额度」里连接",
        Provider::Aliyun => "还没有阿里云百炼密钥，点下方「填入密钥」",
    }
}

fn load() -> Vec<Message> {
    let Some(path) = store() else { return Vec::new() };
    let Ok(raw) = std::fs::read_to_string(path) else { return Vec::new() };
    serde_json::from_str::<Vec<Message>>(&raw).unwrap_or_default()
}

impl Chat {
    fn save(&self) {
        let Some(path) = store() else { return };
        let Some(dir) = path.parent() else { return };
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        if let Ok(raw) = serde_json::to_string(&self.messages) {
            let _ = std::fs::write(path, raw);
        }
    }

    fn busy(&self) -> bool {
        self.streaming
    }

    fn last_assistant(&self) -> Option<&Message> {
        self.messages.iter().rev().find(|m| m.role == Role::Assistant && !m.text.is_empty())
    }
}

/// 从尾部按字符数截取上下文，保证不把一条消息截成半句；首条必须是用户提问。
fn context(messages: &[Message]) -> Vec<Message> {
    let mut picked: Vec<Message> = Vec::new();
    let mut chars = 0usize;
    for message in messages.iter().rev() {
        if picked.len() >= MAX_MESSAGES {
            break;
        }
        let length = message.text.chars().count();
        if chars + length > MAX_CONTEXT_CHARS && !picked.is_empty() {
            break;
        }
        chars += length;
        picked.push(message.clone());
    }
    picked.reverse();
    while picked.first().is_some_and(|m| m.role == Role::Assistant) {
        picked.remove(0);
    }
    picked
}

/// 把基地址补成完整的对话接口。三种写法都认：
/// `https://host`、`https://host/compatible-mode/v1`、`https://host/.../chat/completions`。
fn chat_endpoint(base: &str) -> String {
    let base = base.trim().trim_end_matches('/');
    if base.is_empty() {
        return format!("{ALIYUN_BASE}/compatible-mode/v1/chat/completions");
    }
    if base.ends_with("/chat/completions") {
        base.to_string()
    } else if base.ends_with("/compatible-mode/v1") {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/compatible-mode/v1/chat/completions")
    }
}

/// 模型清单的地址：与对话接口同一套基地址规则。
fn models_endpoint(provider: Provider, base: &str) -> String {
    match provider {
        Provider::DeepSeek => "https://api.deepseek.com/models".to_string(),
        Provider::Aliyun => {
            let base = base.trim().trim_end_matches('/');
            if base.is_empty() {
                format!("{ALIYUN_BASE}/compatible-mode/v1/models")
            } else if let Some(root) = base.strip_suffix("/chat/completions") {
                format!("{root}/models")
            } else if base.ends_with("/compatible-mode/v1") {
                format!("{base}/models")
            } else {
                format!("{base}/compatible-mode/v1/models")
            }
        }
    }
}

/// 结尾是 `-2025-01-25` 这种日期快照编号。
fn ends_with_date(id: &str) -> bool {
    let bytes = id.as_bytes();
    let n = bytes.len();
    if n < 11 {
        return false;
    }
    let tail = &bytes[n - 11..];
    tail[0] == b'-'
        && tail[1..5].iter().all(u8::is_ascii_digit)
        && tail[5] == b'-'
        && tail[6..8].iter().all(u8::is_ascii_digit)
        && tail[8] == b'-'
        && tail[9..11].iter().all(u8::is_ascii_digit)
}

/// 结尾是 `-0107` 这种四位快照编号（qwen-max-0107、deepseek-v4-pro-0813）。
fn ends_with_snapshot(id: &str) -> bool {
    match id.rsplit_once('-') {
        Some((_, tail)) => tail.len() == 4 && tail.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

/// 只有能对话的模型才进面板：语音、图像、向量、重排这些调了只会报错。
fn is_chat_model(id: &str) -> bool {
    let id = id.trim();
    if id.is_empty() || id.len() > 96 || id.chars().any(char::is_whitespace) {
        return false;
    }
    let lower = id.to_ascii_lowercase();
    const SKIP: [&str; 13] = [
        "tts", "asr", "speech", "image", "embedding", "rerank", "realtime", "livetranslate", "wan2",
        "z-image", "gui-plus", "audio", "-ocr",
    ];
    !SKIP.iter().any(|key| lower.contains(key))
}

/// 解析 `/models` 的返回；同一批里保留顺序、去重，并把人造快照排到后面。
fn parse_models(raw: &str) -> Result<Vec<String>, String> {
    let value: serde_json::Value =
        serde_json::from_str(raw).map_err(|_| "模型列表格式异常，稍后再试".to_string())?;
    let Some(list) = value.get("data").and_then(|data| data.as_array()).or_else(|| value.as_array()) else {
        return Err("模型列表格式异常，稍后再试".into());
    };
    let mut named = 0usize;
    let mut models: Vec<String> = Vec::new();
    for entry in list {
        let id = entry.get("id").and_then(|id| id.as_str()).or_else(|| entry.as_str());
        let Some(id) = id else { continue };
        named += 1;
        if !is_chat_model(id) {
            continue;
        }
        if !models.iter().any(|kept| kept == id) {
            models.push(id.to_string());
        }
    }
    if models.is_empty() {
        return Err(if named == 0 {
            "接口没有返回模型列表".into()
        } else {
            format!("接口只有 {named} 个模型，且都不是对话模型")
        });
    }
    models.sort_by(|a, b| {
        let snapshot = |id: &String| ends_with_date(id) || ends_with_snapshot(id);
        snapshot(a).cmp(&snapshot(b)).then_with(|| a.to_ascii_lowercase().cmp(&b.to_ascii_lowercase()))
    });
    Ok(models)
}

/// 后台拉一次模型清单。
fn fetch_models(endpoint: String, key: String, tx: mpsc::Sender<Result<Vec<String>, String>>) {
    std::thread::spawn(move || {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(15))
            .timeout_read(Duration::from_secs(30))
            .build();
        let result = match agent.get(&endpoint).set("Authorization", &format!("Bearer {key}")).call() {
            Ok(response) => {
                let mut raw = String::new();
                match response.into_reader().take(2 * 1024 * 1024).read_to_string(&mut raw) {
                    Ok(_) => parse_models(&raw),
                    Err(_) => Err("读取模型列表失败，稍后再试".into()),
                }
            }
            Err(ureq::Error::Status(401, _)) => Err("密钥无效（401），重新填一次 API Key".into()),
            Err(ureq::Error::Status(code, _)) => Err(format!("模型列表接口返回 {code}")),
            Err(_) => Err("连不上模型列表接口，检查网络后重试".into()),
        };
        let _ = tx.send(result);
    });
}

fn stream(endpoint: String, provider: Provider, key: String, model: String, history: Vec<Message>, tx: mpsc::Sender<Event>, stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(15))
            .timeout_read(Duration::from_secs(90))
            .timeout_write(Duration::from_secs(30))
            .redirects(0)
            .build();
        let mut messages = vec![serde_json::json!({"role": "system", "content": SYSTEM_PROMPT})];
        for message in context(&history) {
            messages.push(serde_json::json!({
                "role": if message.role == Role::User { "user" } else { "assistant" },
                "content": message.text,
            }));
        }
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "temperature": 0.7,
        })
        .to_string();
        let response = match agent
            .post(&endpoint)
            .set("Authorization", &format!("Bearer {key}"))
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream")
            .send_string(&body)
        {
            Ok(response) => response,
            Err(ureq::Error::Status(_, response)) => response,
            Err(_) => {
                let _ = tx.send(Event::Failed(format!("连不上{}，请检查网络后重试", provider.label())));
                return;
            }
        };
        let status = response.status();
        if !(200..300).contains(&status) {
            let mut raw = String::new();
            let _ = response.into_reader().take(64 * 1024).read_to_string(&mut raw);
            let value: serde_json::Value = serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null);
            let detail = value.pointer("/error/message").and_then(|v| v.as_str()).unwrap_or_default().trim().to_string();
            let _ = tx.send(Event::Failed(match status {
                401 => match provider {
                    Provider::DeepSeek => "密钥无效（401），请在「AI 额度」里重新连接".into(),
                    Provider::Aliyun => "百炼密钥无效（401），点标题切到阿里云后重新读入密钥".to_string(),
                },
                402 => format!("{} 额度不足（402）", provider.label()),
                403 => format!("{} 拒绝访问（403）：密钥没开通该模型，或欠费被停", provider.label()),
                429 => "请求太频繁（429），稍后再试".into(),
                _ if detail.is_empty() => format!("请求失败（{status}）"),
                _ => format!("请求失败（{status}）：{detail}"),
            }));
            return;
        }
        let mut reader = BufReader::new(response.into_reader());
        let mut line = String::new();
        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let Some(data) = line.trim_end().strip_prefix("data:") else { continue };
            let data = data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                break;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else { continue };
            if let Some(delta) = value.pointer("/choices/0/delta/content").and_then(|v| v.as_str()) {
                if !delta.is_empty() && tx.send(Event::Delta(delta.to_string())).is_err() {
                    return;
                }
            }
            if let Some(error) = value.pointer("/error/message").and_then(|v| v.as_str()) {
                let _ = tx.send(Event::Failed(error.to_string()));
                return;
            }
        }
        let _ = tx.send(Event::Done);
    });
}

/// 一条消息：用户靠右蓝色，助手靠左灰色。
fn bubble(ui: &mut egui::Ui, card: &Card, message: &Message, streaming_hint: bool) {
    let (fill, align, prefix) = match message.role {
        Role::User => (theme::BLUE_DEEP, egui::Layout::right_to_left(egui::Align::TOP), "你"),
        Role::Assistant => (theme::BTN, egui::Layout::left_to_right(egui::Align::TOP), "AI"),
    };
    let text = if message.text.is_empty() && streaming_hint { "…" } else { message.text.as_str() };
    if text.is_empty() {
        return;
    }
    ui.with_layout(align, |ui| {
        egui::Frame::new()
            .fill(fill)
            .corner_radius(9.0)
            .inner_margin(egui::Margin::symmetric(9, 6))
            .show(ui, |ui| {
                ui.set_max_width(268.0);
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(prefix)
                            .font(card.font(9.0, W::Semibold))
                            .color(theme::DIM),
                    );
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(text)
                                .font(card.font(11.0, W::Regular))
                                .color(theme::WHITE),
                        )
                        .wrap(),
                    );
                });
            });
    });
}

impl App {
    /// 当前服务商的密钥：DeepSeek 取「AI 额度」里已启用账号，阿里云百炼读自己的凭据。
    fn chat_key(&self, provider: Provider) -> Option<String> {
        match provider {
            Provider::Aliyun => aliyun_key(),
            Provider::DeepSeek => {
                let account = self
                    .quota
                    .config
                    .accounts
                    .iter()
                    .find(|a| a.enabled && a.provider == crate::quota::Provider::DeepSeek)?;
                let slot = account.slot.as_deref()?;
                crate::quota::deepseek::load_key(slot).ok()
            }
        }
    }

    fn chat_provider(&self) -> Provider {
        Provider::parse(&self.settings.chat_provider)
    }

    fn chat_model(&self, provider: Provider) -> String {
        let configured = match provider {
            Provider::DeepSeek => &self.settings.chat_model,
            Provider::Aliyun => &self.settings.aliyun_model,
        };
        let model = configured.trim();
        if model.is_empty() { provider.default_model().to_string() } else { model.to_string() }
    }

    /// 「模型」面板里能点到的模型：自己挑过的在前，接口拉回来的在后。
    fn chat_models(&self, provider: Provider) -> Vec<String> {
        let raw = match provider {
            Provider::DeepSeek => &self.settings.chat_models_deepseek,
            Provider::Aliyun => &self.settings.chat_models_aliyun,
        };
        let mut models: Vec<String> = Vec::new();
        for part in raw.split(',').map(str::trim).filter(|part| !part.is_empty()) {
            if !models.iter().any(|kept| kept == part) {
                models.push(part.to_string());
            }
        }
        if models.is_empty() {
            models.extend(provider.models_default().split(',').map(str::to_string));
        }
        models
    }

    /// 选中一个模型：写进设置、记到列表最前面，面板随即收起。
    fn chat_pick_model(&mut self, provider: Provider, model: &str) {
        let model = model.trim();
        if model.is_empty() {
            return;
        }
        match provider {
            Provider::DeepSeek => self.settings.chat_model = model.to_string(),
            Provider::Aliyun => self.settings.aliyun_model = model.to_string(),
        }
        let mut models = self.chat_models(provider);
        models.retain(|kept| kept != model);
        models.insert(0, model.to_string());
        models.truncate(MAX_SAVED_MODELS);
        let joined = models.join(",");
        match provider {
            Provider::DeepSeek => self.settings.chat_models_deepseek = joined,
            Provider::Aliyun => self.settings.chat_models_aliyun = joined,
        }
        self.settings.save();
        self.chat.error.clear();
        self.chat.picker = None;
    }

    fn chat_send(&mut self, provider: Provider, key: String, model: String) {
        let text = self.chat.draft.trim().to_string();
        if text.is_empty() {
            self.chat.error = "先输入一句话".into();
            return;
        }
        self.chat.error.clear();
        self.chat.draft.clear();
        self.chat.messages.push(Message { role: Role::User, text });
        self.chat.messages.push(Message { role: Role::Assistant, text: String::new() });
        let history = self.chat.messages[..self.chat.messages.len().saturating_sub(1)].to_vec();
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        self.chat.rx = Some(rx);
        self.chat.stop = Some(stop.clone());
        self.chat.streaming = true;
        self.chat.scroll_bottom = true;
        self.chat.draft.clear();
        let endpoint = provider.endpoint(&self.settings.aliyun_base);
        stream(endpoint, provider, key, model, history, tx, stop);
    }

    fn chat_stop(&mut self) {
        if let Some(stop) = &self.chat.stop {
            stop.store(true, Ordering::Relaxed);
        }
    }

    pub(crate) fn page_chat(&mut self, c: &Card, _env: &Env, cmds: &mut Vec<Cmd>) {
        // ---------- 收取流式输出 ----------
        let mut events = Vec::new();
        if let Some(rx) = &self.chat.rx {
            loop {
                match rx.try_recv() {
                    Ok(event) => {
                        let last = matches!(event, Event::Done | Event::Failed(_));
                        events.push(event);
                        if last {
                            break;
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        events.push(Event::Failed("对话线程意外退出，请重试".into()));
                        break;
                    }
                }
            }
        }
        let mut finished = false;
        for event in events {
            match event {
                Event::Delta(text) => {
                    if let Some(message) = self.chat.messages.last_mut() {
                        message.text.push_str(&text);
                    }
                }
                Event::Done => finished = true,
                Event::Failed(error) => {
                    self.chat.error = error;
                    if self.chat.messages.last().is_some_and(|m| m.role == Role::Assistant && m.text.is_empty()) {
                        self.chat.messages.pop();
                    }
                    finished = true;
                }
            }
        }
        if finished {
            self.chat.rx = None;
            self.chat.stop = None;
            self.chat.streaming = false;
            self.chat.scroll_bottom = true;
            self.chat.save();
        }
        if self.chat.rx.is_some() {
            c.ui.ctx().request_repaint();
        }

        let provider = self.chat_provider();
        let key = self.chat_key(provider);
        let model = self.chat_model(provider);
        let streaming = self.chat.busy();
        let base = self.settings.aliyun_base.clone();

        // ---------- 顶部 ----------
        c.text(44.0, 44.0, "AI 对话", size::BODY, W::Semibold, theme::WHITE, H::Left);
        // 服务商点一下换另一家；模型名点一下打开模型面板。两家都是 OpenAI 兼容接口，切换不影响历史。
        let provider_w = c.text_w(provider.label(), 9.5, W::Regular);
        c.text(112.0, 44.0, provider.label(), 9.5, W::Regular, theme::DIM, H::Left);
        if c.hit(110.0, 34.0, provider_w + 4.0, 18.0, "chat-provider").clicked() && !streaming {
            self.settings.chat_provider = provider.next().key().to_string();
            self.settings.save();
            self.chat.error.clear();
            self.chat.key_prompt = false;
            self.chat.picker = None;
        }
        let model_x = 112.0 + provider_w + 2.0;
        let model_room = (244.0 - model_x).max(46.0);
        // 只画「· 模型名」：标题栏很窄，多一个 ▾ 就会把模型名截短。
        let chip = format!("· {model}");
        c.text(
            model_x,
            44.0,
            &c.fit(&chip, 9.5, W::Regular, model_room),
            9.5,
            W::Regular,
            if self.chat.picker.is_some() { theme::WHITE } else { theme::GRAY },
            H::Left,
        );
        if c.hit(model_x - 2.0, 34.0, model_room + 4.0, 18.0, "chat-model-open").clicked() && !streaming {
            self.chat.picker = match self.chat.picker {
                Some(_) => None,
                None => Some(Picker::new()),
            };
        }

        let has_reply = self.chat.last_assistant().is_some();
        let copied = self.t - self.chat.copied_at < 1.4;
        let (copy, copy_card) = paint::capsule(c, 248.0, 30.0, 52.0, 19.0, "chat-copy", theme::BTN, theme::BTN_HOVER);
        copy_card.text(274.0, 43.0, if copied { "已复制" } else { "复制" }, size::TINY, W::Semibold, if has_reply { theme::WHITE } else { theme::DIM }, H::Center);
        if copy.clicked() && !streaming {
            if let Some(message) = self.chat.last_assistant() {
                cmds.push(Cmd::Copy(message.text.clone()));
                self.chat.copied_at = self.t;
            }
        }

        let (fresh, fresh_card) = paint::capsule(c, 306.0, 30.0, 71.0, 19.0, "chat-new", theme::BTN, theme::BTN_HOVER);
        fresh_card.text(341.0, 43.0, "新对话", size::TINY, W::Semibold, theme::WHITE, H::Center);
        if fresh.clicked() && !streaming {
            self.chat.messages.clear();
            self.chat.error.clear();
            self.chat.save();
            self.chat.scroll_bottom = true;
        }

        // ---------- 消息区 ----------
        // 模型面板展开时把它盖住，避免两层滚动区抢滚轮。
        if self.chat.picker.is_none() {
        c.fill(42.0, 56.0, 335.0, 138.0, 9.0, theme::CARD);
        let body = c.rect(42.0, 56.0, 335.0, 138.0);
        let area = body.shrink2(egui::vec2(5.0, 4.0));
        let mut content = egui::Ui::new(
            c.ui.ctx().clone(),
            egui::Id::new(("chat-messages", c.live)),
            egui::UiBuilder::new()
                .max_rect(area)
                .layer_id(c.painter.layer_id())
                .style(c.ui.style().clone()),
        );
        content.set_clip_rect(body.intersect(c.painter.clip_rect()));
        content.set_opacity(c.painter.opacity());
        if !c.live {
            content.disable();
        }
        content.spacing_mut().item_spacing.y = 6.0;
        content.spacing_mut().scroll.bar_width = 3.0 * c.s;
        content.spacing_mut().scroll.floating = false;
        egui::ScrollArea::vertical()
            .id_salt("chat-scroll")
            .max_height(area.height())
            .auto_shrink([false, false])
            .stick_to_bottom(self.chat.scroll_bottom)
            .show(&mut content, |ui| {
                if self.chat.messages.is_empty() {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("问点什么，或点「粘贴」把剪贴板的内容放进来")
                            .font(c.font(10.5, W::Regular))
                            .color(theme::DIM),
                    );
                }
                let count = self.chat.messages.len();
                for (index, message) in self.chat.messages.iter().enumerate() {
                    bubble(ui, c, message, streaming && index + 1 == count);
                }
            });
        self.chat.scroll_bottom = false;
        }

        // ---------- 输入框 ----------
        c.fill(42.0, 200.0, 262.0, 30.0, 8.0, theme::BTN);
        let input_id = egui::Id::new("chat-input");
        if std::mem::take(&mut self.chat.blur) {
            c.ui.memory_mut(|m| m.surrender_focus(input_id));
        }
        let input_rect = c.rect(42.0, 200.0, 262.0, 30.0);
        let mut input = egui::Ui::new(
            c.ui.ctx().clone(),
            egui::Id::new(("chat-input-area", c.live)),
            egui::UiBuilder::new()
                .max_rect(input_rect)
                .layer_id(c.painter.layer_id())
                .style(c.ui.style().clone()),
        );
        input.set_clip_rect(input_rect.intersect(c.painter.clip_rect()));
        input.set_opacity(c.painter.opacity());
        if !c.live {
            input.disable();
        }
        let mut draft = std::mem::take(&mut self.chat.draft);
        let edit_rect = input_rect.shrink2(egui::vec2(8.0, 4.0));
        let response = input.put(
            edit_rect,
            egui::TextEdit::multiline(&mut draft)
                .id(input_id)
                .frame(false)
                .margin(egui::Margin::ZERO)
                .text_color(theme::WHITE)
                .font(c.font(11.0, W::Regular))
                .hint_text(egui::RichText::new("说点什么…").color(theme::DIM).font(c.font(11.0, W::Regular)))
                .desired_width(edit_rect.width())
                .desired_rows(2),
        );
        self.chat.draft = draft;
        self.chat.focused = response.has_focus();
        // 岛体是「点击不激活」的悬浮窗：输入框拿到焦点时必须把窗口真正激活，
        // 否则键盘事件仍旧发给原来那个前台程序，用户看到的就是「打不出字」。
        if response.gained_focus() {
            cmds.push(Cmd::Focus);
        }
        let enter = input.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
        if enter && response.has_focus() && !streaming {
            self.chat.draft = self.chat.draft.trim_end().to_string();
            if let Some(key) = key.clone() {
                self.chat_send(provider, key, model.clone());
            } else {
                self.chat.error = missing_key(provider).into();
            }
            response.request_focus();
        }

        // ---------- 发送 / 停止 ----------
        let can_send = !streaming && !self.chat.draft.trim().is_empty();
        let (go, go_card) = paint::capsule(
            c,
            310.0,
            200.0,
            67.0,
            30.0,
            "chat-send",
            if streaming || can_send { theme::BLUE_DEEP } else { theme::BTN },
            if streaming || can_send { theme::BLUE } else { theme::BTN },
        );
        go_card.text(
            343.0,
            219.0,
            if streaming { "停止" } else { "发送" },
            size::BODY,
            W::Semibold,
            if streaming || can_send { theme::WHITE } else { theme::DIM },
            H::Center,
        );
        if go.clicked() {
            if streaming {
                self.chat_stop();
            } else if let Some(key) = key.clone() {
                self.chat_send(provider, key, model.clone());
            } else {
                self.chat.error = missing_key(provider).into();
            }
        }

        // ---------- 粘贴 ----------
        let (paste, paste_card) = paint::capsule(c, 42.0, 232.0, 48.0, 17.0, "chat-paste", theme::BTN, theme::BTN_HOVER);
        paste_card.text(66.0, 243.0, "粘贴", size::TINY, W::Semibold, theme::WHITE, H::Center);
        if paste.clicked() && !streaming {
            if let Some(text) = crate::clipboard::current_text() {
                if !text.trim().is_empty() {
                    self.chat.draft = text;
                }
            }
        }

        // ---------- 底部提示 ----------
        if !self.chat.error.is_empty() {
            c.text(96.0, 243.0, &c.fit(&self.chat.error, 9.0, W::Regular, 200.0), 9.0, W::Regular, theme::RED_TOP, H::Left);
        } else if key.is_none() {
            match provider {
                Provider::DeepSeek => {
                    c.text(96.0, 243.0, "先在「AI 额度」里连接 DeepSeek API Key", 9.0, W::Regular, theme::DIM, H::Left);
                    self.chat_connect_link(c);
                }
                Provider::Aliyun if self.chat.key_prompt => {
                    c.text(96.0, 243.0, "复制百炼 API Key，再点「读取」", 9.0, W::Regular, theme::DIM, H::Left);
                    let (read, read_card) = paint::capsule(c, 300.0, 232.0, 57.0, 17.0, "chat-key-read", theme::BLUE_DEEP, theme::BLUE);
                    read_card.text(328.0, 243.0, "读取", size::TINY, W::Semibold, theme::WHITE, H::Center);
                    if read.clicked() {
                        let pasted = crate::clipboard::current_text().unwrap_or_default();
                        match save_aliyun_key(&pasted) {
                            Ok(()) => {
                                self.chat.key_prompt = false;
                                self.chat.key_saved_at = self.t;
                            }
                            Err(error) => self.chat.error = error,
                        }
                    }
                }
                Provider::Aliyun => {
                    c.text(96.0, 243.0, "还没有阿里云百炼 API Key", 9.0, W::Regular, theme::DIM, H::Left);
                    let width = c.text_w("还没有阿里云百炼 API Key", 9.0, W::Regular);
                    let x = 96.0 + width + 6.0;
                    c.text(x, 243.0, "填入密钥", 9.0, W::Semibold, theme::BLUE, H::Left);
                    if c.hit(x - 2.0, 232.0, 60.0, 16.0, "chat-key-prompt").clicked() {
                        self.chat.key_prompt = true;
                    }
                }
            }
        } else if self.t - self.chat.key_saved_at < 1.6 {
            c.text(96.0, 243.0, "密钥已保存，可以提问了", 9.0, W::Regular, theme::GREEN, H::Left);
        } else if streaming {
            c.text(96.0, 243.0, "正在回答…", 9.0, W::Regular, theme::BLUE, H::Left);
        } else {
            c.text(
                96.0,
                243.0,
                "Enter 发送 · Shift+Enter 换行 · 点模型名换模型",
                9.0,
                W::Regular,
                theme::DIM,
                H::Left,
            );
        }

        // ---------- 模型面板（最后画，盖在消息区上面） ----------
        if self.chat.picker.is_some() {
            self.chat_model_panel(c, provider, key, &base, cmds);
        }
    }

    /// 「模型」面板：筛选框 + 从接口拉全部模型 + 点一个就切换。
    fn chat_model_panel(&mut self, c: &Card, provider: Provider, key: Option<String>, base: &str, cmds: &mut Vec<Cmd>) {
        let Some(mut picker) = self.chat.picker.take() else { return };
        // 后台拉取的结果
        if let Some(rx) = self.chat.catalog_rx.take() {
            match rx.try_recv() {
                Ok(Ok(models)) => {
                    self.chat.catalog = models;
                    self.chat.catalog_loading = false;
                    self.chat.catalog_error.clear();
                }
                Ok(Err(error)) => {
                    self.chat.catalog_error = error;
                    self.chat.catalog_loading = false;
                }
                Err(TryRecvError::Empty) => self.chat.catalog_rx = Some(rx),
                Err(TryRecvError::Disconnected) => {
                    self.chat.catalog_error = "模型列表线程意外退出，再点一次「拉取」".into();
                    self.chat.catalog_loading = false;
                }
            }
        }
        // 第一次打开面板就自动拉一遍，不用先点按钮
        if !self.chat.catalog_tried && !self.chat.catalog_loading && self.chat.catalog_rx.is_none() {
            if let Some(key) = key.clone() {
                let (tx, rx) = mpsc::channel();
                fetch_models(models_endpoint(provider, base), key, tx);
                self.chat.catalog_rx = Some(rx);
                self.chat.catalog_loading = true;
                self.chat.catalog_tried = true;
            }
        }
        if self.chat.catalog_loading {
            c.ui.ctx().request_repaint();
        }

        let current = self.chat_model(provider);
        let saved = self.chat_models(provider);
        let mut close = c.ui.input(|input| input.key_pressed(egui::Key::Escape));

        // 底板
        c.fill(42.0, 56.0, 335.0, 138.0, 9.0, theme::CARD);
        c.stroke(42.0, 56.0, 335.0, 138.0, 9.0, 1.0, paint::alpha(theme::BTN_HOVER, 0.7));

        // 筛选框
        let field = c.rect(48.0, 60.0, 188.0, 20.0);
        c.fill(48.0, 60.0, 188.0, 20.0, 6.0, theme::BTN);
        let mut filter_ui = egui::Ui::new(
            c.ui.ctx().clone(),
            egui::Id::new(("chat-model-filter", c.live)),
            egui::UiBuilder::new()
                .max_rect(field)
                .layer_id(c.painter.layer_id())
                .style(c.ui.style().clone()),
        );
        filter_ui.set_clip_rect(field.intersect(c.painter.clip_rect()));
        filter_ui.set_opacity(c.painter.opacity());
        if !c.live {
            filter_ui.disable();
        }
        let filter_response = filter_ui.put(
            field.shrink2(egui::vec2(6.0, 3.0)),
            egui::TextEdit::singleline(&mut picker.filter)
                .id(egui::Id::new("chat-model-filter-edit"))
                .frame(false)
                .margin(egui::Margin::ZERO)
                .text_color(theme::WHITE)
                .font(c.font(10.5, W::Regular))
                .hint_text(
                    egui::RichText::new("筛选模型…")
                        .color(theme::DIM)
                        .font(c.font(10.5, W::Regular)),
                )
                .desired_width(field.width() - 12.0),
        );
        if filter_response.gained_focus() {
            cmds.push(Cmd::Focus);
        }

        // 「拉取」按钮
        let (refresh, refresh_card) = paint::capsule(c, 242.0, 60.0, 46.0, 20.0, "chat-model-refresh", theme::BTN, theme::BTN_HOVER);
        refresh_card.text(265.0, 73.0, if self.chat.catalog_loading { "拉取中" } else { "拉取" }, size::TINY, W::Semibold, if self.chat.catalog_loading { theme::DIM } else { theme::WHITE }, H::Center);
        if refresh.clicked() && !self.chat.catalog_loading {
            match key.clone() {
                Some(key) => {
                    let (tx, rx) = mpsc::channel();
                    fetch_models(models_endpoint(provider, base), key, tx);
                    self.chat.catalog_rx = Some(rx);
                    self.chat.catalog_loading = true;
                    self.chat.catalog_tried = true;
                    self.chat.catalog_error.clear();
                }
                None => self.chat.catalog_error = missing_key(provider).into(),
            }
        }

        // 关闭
        let close_button = c.hit(345.0, 60.0, 22.0, 20.0, "chat-model-close");
        paint::close(c, 356.0, 70.0, 4.2, if close_button.hovered() { theme::WHITE } else { theme::DIM });
        if close_button.clicked() {
            close = true;
        }

        // 列表：自己挑过的在前，接口拉回来的在后
        let needle = picker.filter.trim().to_ascii_lowercase();
        let mut pool: Vec<String> = saved.clone();
        for id in &self.chat.catalog {
            if !pool.iter().any(|kept| kept == id) {
                pool.push(id.clone());
            }
        }
        let visible: Vec<String> = pool
            .into_iter()
            .filter(|id| needle.is_empty() || id.to_ascii_lowercase().contains(&needle))
            .collect();
        if !self.chat.catalog_error.is_empty() {
            c.text(48.0, 88.0, &c.fit(&self.chat.catalog_error, 9.0, W::Regular, 320.0), 9.0, W::Regular, theme::RED_TOP, H::Left);
        }
        let list = c.rect(46.0, 84.0, 331.0, 106.0);
        let mut list_ui = egui::Ui::new(
            c.ui.ctx().clone(),
            egui::Id::new(("chat-model-list", c.live)),
            egui::UiBuilder::new()
                .max_rect(list)
                .layer_id(c.painter.layer_id())
                .style(c.ui.style().clone()),
        );
        list_ui.set_clip_rect(list.intersect(c.painter.clip_rect()));
        list_ui.set_opacity(c.painter.opacity());
        if !c.live {
            list_ui.disable();
        }
        let font = c.font(10.5, W::Regular);
        let mut chosen: Option<String> = None;
        egui::ScrollArea::vertical()
            .id_salt("chat-model-scroll")
            .max_height(list.height())
            .auto_shrink([false, false])
            .show(&mut list_ui, |ui| {
                if visible.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new(if self.chat.catalog_loading { "正在拉取模型列表…" } else if needle.is_empty() { "这里还是空的，点「拉取」从接口取" } else { "没有匹配的模型" })
                            .font(c.font(10.0, W::Regular))
                            .color(theme::DIM),
                    );
                }
                for id in &visible {
                    let selected = id.as_str() == current;
                    let pinned = saved.iter().any(|kept| kept == id);
                    let (row, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 19.0), egui::Sense::click());
                    if selected {
                        ui.painter().rect_filled(row, 5.0, theme::BLUE_DEEP);
                    } else if response.hovered() {
                        ui.painter().rect_filled(row, 5.0, theme::BTN);
                    }
                    let label = if selected {
                        format!("✓ {id}")
                    } else if pinned {
                        format!("· {id}")
                    } else {
                        id.clone()
                    };
                    ui.painter().text(
                        row.left_center() + egui::vec2(7.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        &c.fit(&label, 10.5, W::Regular, row.width() - 14.0),
                        font.clone(),
                        if selected { theme::WHITE } else if pinned { theme::GRAY } else { theme::DIM },
                    );
                    if response.clicked() {
                        chosen = Some(id.clone());
                    }
                }
            });

        self.chat.focused = self.chat.focused || filter_response.has_focus();
        match chosen {
            Some(model) => self.chat_pick_model(provider, &model),
            None if close => self.chat.picker = None,
            None => self.chat.picker = Some(picker),
        }
    }

    /// 没有密钥时，在提示文字后面接一个跳转到「AI 额度」的链接。
    fn chat_connect_link(&mut self, c: &Card) {
        let width = c.text_w("先在「AI 额度」里连接 DeepSeek API Key", 9.0, W::Regular);
        let x = 96.0 + width + 6.0;
        c.text(x, 243.0, "去连接", 9.0, W::Semibold, theme::BLUE, H::Left);
        if c.hit(x - 2.0, 232.0, 46.0, 16.0, "chat-connect").clicked() {
            self.settings_tab = PAGE_SETTINGS_TABS[QUOTA_PAGE];
            self.open_settings();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(role: Role, text: &str) -> Message {
        Message { role, text: text.to_string() }
    }

    #[test]
    fn context_drops_leading_assistant_and_keeps_recent() {
        let messages = vec![
            message(Role::Assistant, "旧回复"),
            message(Role::User, "问题一"),
            message(Role::Assistant, "回答一"),
            message(Role::User, "问题二"),
        ];
        let picked = context(&messages);
        assert_eq!(picked.len(), 3);
        assert_eq!(picked[0].role, Role::User);
        assert_eq!(picked.last().unwrap().text, "问题二");
    }

    #[test]
    fn context_stops_at_character_budget() {
        let long = "字".repeat(MAX_CONTEXT_CHARS);
        let messages = vec![message(Role::User, &long), message(Role::Assistant, "短"), message(Role::User, "最新")];
        let picked = context(&messages);
        assert!(picked.len() < messages.len(), "超长历史应被截断");
        assert_eq!(picked.last().unwrap().text, "最新");
    }

    #[test]
    fn context_keeps_single_huge_message() {
        let long = "字".repeat(MAX_CONTEXT_CHARS + 500);
        let picked = context(&[message(Role::User, &long)]);
        assert_eq!(picked.len(), 1, "只有一条消息时不该被截空");
    }

    #[test]
    fn empty_assistant_placeholder_is_not_reported_as_reply() {
        let chat = Chat {
            messages: vec![message(Role::User, "你好"), message(Role::Assistant, "")],
            ..Chat::default()
        };
        assert!(chat.last_assistant().is_none());
    }

    #[test]
    fn base_url_fills_the_compatible_chat_path() {
        // 空值走官方地址
        assert_eq!(chat_endpoint(""), "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions");
        // 只给域名（含末尾斜杠）也能补齐
        assert_eq!(chat_endpoint("https://maas.example.com/"), "https://maas.example.com/compatible-mode/v1/chat/completions");
        // 已经写到 /compatible-mode/v1
        assert_eq!(chat_endpoint("https://maas.example.com/compatible-mode/v1"), "https://maas.example.com/compatible-mode/v1/chat/completions");
        // 已经写全，原样使用
        assert_eq!(chat_endpoint("https://maas.example.com/compatible-mode/v1/chat/completions"), "https://maas.example.com/compatible-mode/v1/chat/completions");
        // DeepSeek 不受基地址影响
        assert_eq!(Provider::DeepSeek.endpoint("https://maas.example.com"), "https://api.deepseek.com/chat/completions");
    }

    #[test]
    fn models_endpoint_follows_the_same_base_rules() {
        assert_eq!(models_endpoint(Provider::Aliyun, ""), "https://dashscope.aliyuncs.com/compatible-mode/v1/models");
        assert_eq!(models_endpoint(Provider::Aliyun, "https://maas.example.com"), "https://maas.example.com/compatible-mode/v1/models");
        assert_eq!(models_endpoint(Provider::Aliyun, "https://maas.example.com/compatible-mode/v1"), "https://maas.example.com/compatible-mode/v1/models");
        assert_eq!(
            models_endpoint(Provider::Aliyun, "https://maas.example.com/compatible-mode/v1/chat/completions"),
            "https://maas.example.com/compatible-mode/v1/models"
        );
        assert_eq!(models_endpoint(Provider::DeepSeek, "https://maas.example.com"), "https://api.deepseek.com/models");
    }

    #[test]
    fn non_chat_models_are_kept_out_of_the_panel() {
        for id in ["qwen-plus", "deepseek-v4.1-flash", "glm-5.3", "kimi-k3", "qwen3-vl-plus"] {
            assert!(is_chat_model(id), "{id} 应该能对话");
        }
        for id in ["qwen3-tts-flash", "fun-asr-flash-2026-06-15", "qwen-image-3.0", "text-embedding-v4", "qwen3.7-text-rerank", "wan2.7-image", "z-image-turbo", "qwen-vl-ocr"] {
            assert!(!is_chat_model(id), "{id} 不是对话模型");
        }
        assert!(!is_chat_model(""));
        assert!(!is_chat_model("has space"));
    }

    #[test]
    fn snapshot_ids_sort_after_plain_aliases() {
        assert!(ends_with_date("qwen-plus-2025-01-25"));
        assert!(!ends_with_date("qwen-plus"));
        assert!(ends_with_snapshot("qwen-max-0107"));
        assert!(!ends_with_snapshot("qwen3.5-397b-a17b"));
        let raw = r#"{"data":[{"id":"qwen-plus-2025-01-25"},{"id":"qwen-plus"},{"id":"qwen3-tts-flash"},{"id":"qwen-plus"},{"id":"glm-5.3"}]}"#;
        let models = parse_models(raw).unwrap();
        assert_eq!(models, vec!["glm-5.3", "qwen-plus", "qwen-plus-2025-01-25"], "去重、按名字排，快照排在普通别名之后");
    }

    #[test]
    fn model_list_without_chat_models_is_an_error() {
        let raw = r#"{"data":[{"id":"qwen3-tts-flash"},{"id":"wan2.7-image"}]}"#;
        assert!(parse_models(raw).is_err(), "只有语音和图片模型时要报错，而不是给一个空列表");
        assert!(parse_models("not json").is_err());
    }
}