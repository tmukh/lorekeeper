import {
  App, Component, ItemView, MarkdownRenderer, Notice, Plugin,
  PluginSettingTab, Setting, WorkspaceLeaf, requestUrl, setIcon,
} from "obsidian";
import { spawn, ChildProcess } from "child_process";

interface LoreSettings {
  pythonPath: string;
  repoPath: string;
  port: number;
  autostart: boolean;
}

const DEFAULT_SETTINGS: LoreSettings = {
  pythonPath: "D:\\dev\\lorekeeper\\.venv\\Scripts\\python.exe",
  repoPath: "D:\\dev\\lorekeeper",
  port: 8765,
  autostart: true,
};

const VIEW_TYPE = "lorekeeper-chat";

interface Source { title: string; path: string; }
interface Answer { answer: string; sources: Source[]; }

export default class LorekeeperPlugin extends Plugin {
  settings: LoreSettings = DEFAULT_SETTINGS;
  proc: ChildProcess | null = null;

  async onload() {
    await this.loadSettings();
    this.registerView(VIEW_TYPE, (leaf) => new ChatView(leaf, this));
    this.addRibbonIcon("bot", "Lorekeeper chat", () => this.activateView());
    this.addCommand({ id: "open-lorekeeper-chat", name: "Open Lorekeeper chat", callback: () => this.activateView() });
    this.addSettingTab(new LoreSettingTab(this.app, this));
    if (this.settings.autostart) await this.startServer();
  }

  onunload() {
    this.stopServer();
  }

  baseUrl(): string {
    return `http://127.0.0.1:${this.settings.port}`;
  }

  async startServer() {
    if (this.proc) return;
    try {
      const h = await requestUrl({ url: `${this.baseUrl()}/health`, method: "GET", throw: false });
      if (h.status === 200) {
        new Notice("Lorekeeper: using the server that's already running.");
        return;
      }
    } catch (_) {
      /* not up — spawn below */
    }
    try {
      this.proc = spawn(this.settings.pythonPath, ["-m", "lorekeeper.server", "--rest"], {
        cwd: this.settings.repoPath,
        env: process.env,
      });
      this.proc.stdout?.on("data", (d) => console.log("[lorekeeper]", d.toString().trim()));
      this.proc.stderr?.on("data", (d) => console.log("[lorekeeper]", d.toString().trim()));
      this.proc.on("exit", (code) => {
        console.log("[lorekeeper] server exited", code);
        this.proc = null;
      });
      new Notice("Lorekeeper: starting local server…");
    } catch (e) {
      new Notice("Lorekeeper: couldn't start the server — check the Python path in settings.");
      console.error(e);
    }
  }

  stopServer() {
    if (this.proc) {
      this.proc.kill();
      this.proc = null;
    }
  }

  async activateView() {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(VIEW_TYPE)[0];
    if (!leaf) {
      const right = workspace.getRightLeaf(false);
      if (!right) return;
      leaf = right;
      await leaf.setViewState({ type: VIEW_TYPE, active: true });
    }
    workspace.revealLeaf(leaf);
  }

  async ask(question: string, mode = "hybrid"): Promise<Answer> {
    try {
      const res = await requestUrl({
        url: `${this.baseUrl()}/query`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ question, mode }),
        throw: false,
      });
      if (res.status >= 400) return { answer: `⚠️ server error ${res.status}: ${res.text}`, sources: [] };
      return { answer: res.json?.answer ?? "(no answer)", sources: res.json?.sources ?? [] };
    } catch (e) {
      return { answer: `⚠️ couldn't reach the lorekeeper server at ${this.baseUrl()} — is it running? (${e})`, sources: [] };
    }
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }
  async saveSettings() {
    await this.saveData(this.settings);
  }
}

class ChatView extends ItemView {
  plugin: LorekeeperPlugin;
  constructor(leaf: WorkspaceLeaf, plugin: LorekeeperPlugin) {
    super(leaf);
    this.plugin = plugin;
  }
  getViewType() { return VIEW_TYPE; }
  getDisplayText() { return "Lorekeeper"; }
  getIcon() { return "bot"; }

  async onOpen() {
    const root = this.containerEl.children[1] as HTMLElement;
    root.empty();
    root.addClass("lorekeeper-chat");
    const log = root.createDiv({ cls: "lk-log" });
    const form = root.createDiv({ cls: "lk-form" });
    const input = form.createEl("textarea", {
      cls: "lk-input",
      attr: { rows: "2", placeholder: "Ask your knowledge graph…  (Enter to send, Shift+Enter = newline)" },
    });
    const send = form.createEl("button", { text: "Ask", cls: "lk-send" });

    const addUser = (text: string) => {
      const m = log.createDiv({ cls: "lk-msg lk-you" });
      m.createDiv({ cls: "lk-who", text: "You" });
      m.createDiv({ cls: "lk-text", text });
      log.scrollTop = log.scrollHeight;
    };

    const addBot = async (a: Answer) => {
      const m = log.createDiv({ cls: "lk-msg lk-bot" });
      const head = m.createDiv({ cls: "lk-head" });
      head.createSpan({ cls: "lk-who", text: "Lorekeeper" });
      const copyBtn = head.createEl("button", { cls: "lk-copy", attr: { "aria-label": "Copy answer" } });
      setIcon(copyBtn, "copy");
      copyBtn.onclick = async () => {
        await navigator.clipboard.writeText(a.answer);
        new Notice("Copied");
      };
      const text = m.createDiv({ cls: "lk-text lk-markdown" });
      await MarkdownRenderer.render(this.plugin.app, a.answer, text, "", this as Component);
      if (a.sources && a.sources.length) {
        const src = m.createDiv({ cls: "lk-sources" });
        src.createSpan({ cls: "lk-sources-label", text: "Sources" });
        const chips = src.createDiv({ cls: "lk-chips" });
        for (const s of a.sources) {
          const chip = chips.createEl("a", { cls: "lk-chip", text: s.title });
          chip.onclick = (e) => {
            e.preventDefault();
            this.plugin.app.workspace.openLinkText(s.path, "", false);
          };
        }
      }
      log.scrollTop = log.scrollHeight;
    };

    const submit = async () => {
      const q = input.value.trim();
      if (!q) return;
      addUser(q);
      input.value = "";
      const thinking = log.createDiv({ cls: "lk-msg lk-bot lk-thinking" });
      thinking.createDiv({ cls: "lk-who", text: "Lorekeeper" });
      thinking.createDiv({ cls: "lk-text", text: "…thinking…" });
      log.scrollTop = log.scrollHeight;
      const ans = await this.plugin.ask(q);
      thinking.remove();
      await addBot(ans);
    };

    send.onclick = submit;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    });
  }

  async onClose() {}
}

class LoreSettingTab extends PluginSettingTab {
  plugin: LorekeeperPlugin;
  constructor(app: App, plugin: LorekeeperPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    new Setting(containerEl).setName("Python path").setDesc("Path to the lorekeeper venv's python.exe")
      .addText((t) => t.setValue(this.plugin.settings.pythonPath).onChange(async (v) => { this.plugin.settings.pythonPath = v.trim(); await this.plugin.saveSettings(); }));
    new Setting(containerEl).setName("Repo path").setDesc("The lorekeeper repo folder (working dir, so .env loads)")
      .addText((t) => t.setValue(this.plugin.settings.repoPath).onChange(async (v) => { this.plugin.settings.repoPath = v.trim(); await this.plugin.saveSettings(); }));
    new Setting(containerEl).setName("Server port")
      .addText((t) => t.setValue(String(this.plugin.settings.port)).onChange(async (v) => { this.plugin.settings.port = parseInt(v) || 8765; await this.plugin.saveSettings(); }));
    new Setting(containerEl).setName("Auto-start server with Obsidian")
      .addToggle((t) => t.setValue(this.plugin.settings.autostart).onChange(async (v) => { this.plugin.settings.autostart = v; await this.plugin.saveSettings(); }));
    new Setting(containerEl).setName("Restart server now")
      .addButton((b) => b.setButtonText("Restart").onClick(async () => { this.plugin.stopServer(); await this.plugin.startServer(); new Notice("Lorekeeper server restarting…"); }));
  }
}
