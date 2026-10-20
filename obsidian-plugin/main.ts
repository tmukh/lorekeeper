import { App, ItemView, Notice, Plugin, PluginSettingTab, Setting, WorkspaceLeaf, requestUrl } from "obsidian";
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
    // Reuse an already-running server (e.g. a manual --rest, or Claude Desktop's instance on this port).
    try {
      const h = await requestUrl({ url: `${this.baseUrl()}/health`, method: "GET", throw: false });
      if (h.status === 200) {
        new Notice("Lorekeeper: using the server that's already running.");
        return;
      }
    } catch (_) {
      /* not up yet — we'll spawn it */
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

  async ask(question: string, mode = "hybrid"): Promise<string> {
    try {
      const res = await requestUrl({
        url: `${this.baseUrl()}/query`,
        method: "POST",
        contentType: "application/json",
        body: JSON.stringify({ question, mode }),
        throw: false,
      });
      if (res.status >= 400) return `⚠️ server error ${res.status}: ${res.text}`;
      return res.json?.answer ?? "(no answer)";
    } catch (e) {
      return `⚠️ couldn't reach the lorekeeper server at ${this.baseUrl()} — is it running? (${e})`;
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
  getViewType() {
    return VIEW_TYPE;
  }
  getDisplayText() {
    return "Lorekeeper";
  }
  getIcon() {
    return "bot";
  }

  async onOpen() {
    const root = this.containerEl.children[1] as HTMLElement;
    root.empty();
    root.addClass("lorekeeper-chat");
    const log = root.createDiv({ cls: "lk-log" });
    const form = root.createDiv({ cls: "lk-form" });
    const input = form.createEl("textarea", {
      cls: "lk-input",
      attr: { rows: "2", placeholder: "Ask your knowledge graph…  (Enter to send, Shift+Enter for newline)" },
    });
    const send = form.createEl("button", { text: "Ask", cls: "lk-send" });

    const addMsg = (who: "you" | "bot", text: string): HTMLElement => {
      const m = log.createDiv({ cls: `lk-msg lk-${who}` });
      m.createDiv({ cls: "lk-who", text: who === "you" ? "You" : "Lorekeeper" });
      m.createDiv({ cls: "lk-text", text });
      log.scrollTop = log.scrollHeight;
      return m;
    };

    const submit = async () => {
      const q = input.value.trim();
      if (!q) return;
      addMsg("you", q);
      input.value = "";
      const thinking = addMsg("bot", "…thinking…");
      const ans = await this.plugin.ask(q);
      thinking.remove();
      addMsg("bot", ans);
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
    new Setting(containerEl)
      .setName("Python path")
      .setDesc("Path to the lorekeeper venv's python.exe")
      .addText((t) =>
        t.setValue(this.plugin.settings.pythonPath).onChange(async (v) => {
          this.plugin.settings.pythonPath = v.trim();
          await this.plugin.saveSettings();
        })
      );
    new Setting(containerEl)
      .setName("Repo path")
      .setDesc("The lorekeeper repo folder (used as working dir so .env loads)")
      .addText((t) =>
        t.setValue(this.plugin.settings.repoPath).onChange(async (v) => {
          this.plugin.settings.repoPath = v.trim();
          await this.plugin.saveSettings();
        })
      );
    new Setting(containerEl)
      .setName("Server port")
      .addText((t) =>
        t.setValue(String(this.plugin.settings.port)).onChange(async (v) => {
          this.plugin.settings.port = parseInt(v) || 8765;
          await this.plugin.saveSettings();
        })
      );
    new Setting(containerEl)
      .setName("Auto-start server with Obsidian")
      .addToggle((t) =>
        t.setValue(this.plugin.settings.autostart).onChange(async (v) => {
          this.plugin.settings.autostart = v;
          await this.plugin.saveSettings();
        })
      );
    new Setting(containerEl)
      .setName("Restart server now")
      .addButton((b) =>
        b.setButtonText("Restart").onClick(async () => {
          this.plugin.stopServer();
          await this.plugin.startServer();
          new Notice("Lorekeeper server restarting…");
        })
      );
  }
}
