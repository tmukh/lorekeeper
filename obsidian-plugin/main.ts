import {
  App, Component, ItemView, MarkdownRenderer, Modal, Notice, Plugin,
  PluginSettingTab, Setting, TFile, WorkspaceLeaf, requestUrl, setIcon,
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
interface Plan {
  folder: string; folder_exists: boolean; is_new: boolean;
  filename: string; template: string | null;
  tags: string[]; links: string[]; reasoning: string;
}

function validFolder(f: string): boolean {
  return !!f && f !== "?" && f !== "/" && !/[<>:"|?*]/.test(f);
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "untitled";
}

function renderPlanPreview(el: HTMLElement, plan: Plan) {
  el.empty();
  const line = (k: string, v: string) => {
    const d = el.createDiv({ cls: "lk-plan-line" });
    d.createSpan({ cls: "lk-plan-k", text: k });
    d.createSpan({ cls: "lk-plan-v", text: v });
  };
  line("Folder", plan.folder + (plan.is_new ? "  (new)" : "  (exists)"));
  line("File", `${plan.folder}/${plan.filename}.md`);
  line("Template", plan.template || "(none)");
  line("Tags", (plan.tags || []).map((t) => "#" + t).join("  ") || "(none)");
  line("Links", (plan.links || []).map((l) => `[[${l}]]`).join(", ") || "(none)");
  if (plan.reasoning) line("Why", plan.reasoning);
}

export default class LorekeeperPlugin extends Plugin {
  settings: LoreSettings = DEFAULT_SETTINGS;
  proc: ChildProcess | null = null;

  async onload() {
    await this.loadSettings();
    this.registerView(VIEW_TYPE, (leaf) => new ChatView(leaf, this));
    this.addRibbonIcon("bot", "Lorekeeper chat", () => this.activateView());
    this.addCommand({ id: "open-lorekeeper-chat", name: "Open Lorekeeper chat", callback: () => this.activateView() });
    this.addCommand({ id: "new-smart-note", name: "New smart note", callback: () => new SmartNoteModal(this.app, this).open() });
    this.addCommand({
      id: "file-this-note",
      name: "File this note (place + tag)",
      callback: async () => {
        const file = this.app.workspace.getActiveFile();
        if (!file) { new Notice("No active note."); return; }
        const content = await this.app.vault.read(file);
        const body = content.replace(/^---\n[\s\S]*?\n---\n/, "");
        new Notice("Lorekeeper: planning…");
        const plan = await this.plan(body || file.basename, file.basename);
        if (plan) new PlanConfirmModal(this.app, plan, `File "${file.basename}"`, "File it", async () => this.fileNote(file, plan)).open();
      },
    });
    this.addCommand({ id: "reindex-vault", name: "Reindex vault", callback: () => this.reindex() });
    this.addSettingTab(new LoreSettingTab(this.app, this));
    if (this.settings.autostart) await this.startServer();
  }

  onunload() { this.stopServer(); }

  baseUrl(): string { return `http://127.0.0.1:${this.settings.port}`; }

  async startServer() {
    if (this.proc) return;
    try {
      const h = await requestUrl({ url: `${this.baseUrl()}/health`, method: "GET", throw: false });
      if (h.status === 200) { new Notice("Lorekeeper: using the server that's already running."); return; }
    } catch (_) { /* spawn below */ }
    try {
      this.proc = spawn(this.settings.pythonPath, ["-m", "lorekeeper.server", "--rest"], { cwd: this.settings.repoPath, env: process.env });
      this.proc.stdout?.on("data", (d) => console.log("[lorekeeper]", d.toString().trim()));
      this.proc.stderr?.on("data", (d) => console.log("[lorekeeper]", d.toString().trim()));
      this.proc.on("exit", (code) => { console.log("[lorekeeper] server exited", code); this.proc = null; });
      new Notice("Lorekeeper: starting local server…");
    } catch (e) {
      new Notice("Lorekeeper: couldn't start the server — check the Python path in settings.");
      console.error(e);
    }
  }

  stopServer() { if (this.proc) { this.proc.kill(); this.proc = null; } }

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
        url: `${this.baseUrl()}/query`, method: "POST", contentType: "application/json",
        body: JSON.stringify({ question, mode }), throw: false,
      });
      if (res.status >= 400) return { answer: `⚠️ server error ${res.status}: ${res.text}`, sources: [] };
      return { answer: res.json?.answer ?? "(no answer)", sources: res.json?.sources ?? [] };
    } catch (e) {
      return { answer: `⚠️ couldn't reach the lorekeeper server at ${this.baseUrl()} — is it running? (${e})`, sources: [] };
    }
  }

  async plan(draft: string, title = ""): Promise<Plan | null> {
    try {
      const res = await requestUrl({
        url: `${this.baseUrl()}/suggest`, method: "POST", contentType: "application/json",
        body: JSON.stringify({ draft_text: draft, title }), throw: false,
      });
      if (res.status >= 400) { new Notice(`Lorekeeper: server error ${res.status}`); return null; }
      return (res.json?.plan as Plan) ?? null;
    } catch (e) {
      new Notice(`Lorekeeper: can't reach server (${e})`);
      return null;
    }
  }

  // ---- note operations ----
  getTemplater(): any | null {
    const p: any = (this.app as any).plugins?.plugins?.["templater-obsidian"];
    return p?.templater ?? null;
  }
  templateFile(name: string): TFile | null {
    const f = this.app.vault.getAbstractFileByPath(`Templates/${name}.md`);
    return f instanceof TFile ? f : null;
  }
  lightSubstitute(content: string, title: string): string {
    const m: any = (window as any).moment;
    const today = m ? m().format("YYYY-MM-DD") : new Date().toISOString().slice(0, 10);
    return content
      .replace(/<%\s*tp\.file\.title\s*%>/g, title)
      .replace(/<%\s*tp\.date\.now\([^)]*\)\s*%>/g, today)
      .replace(/{{\s*title\s*}}/g, title)
      .replace(/{{\s*date\s*}}/g, today);
  }
  async ensureFolder(path: string) {
    const parts = path.split("/");
    let cur = "";
    for (const p of parts) {
      cur = cur ? `${cur}/${p}` : p;
      if (!this.app.vault.getAbstractFileByPath(cur)) {
        try { await this.app.vault.createFolder(cur); } catch (_) { /* exists / race */ }
      }
    }
  }
  async applyTagsAndLinks(file: TFile, plan: Plan) {
    if (plan.tags && plan.tags.length) {
      try {
        await this.app.fileManager.processFrontMatter(file, (fm: any) => {
          const cur: string[] = Array.isArray(fm.tags) ? fm.tags : (fm.tags ? [fm.tags] : []);
          const set = new Set<string>(cur.map((t) => String(t)));
          for (const t of plan.tags) set.add(t);
          fm.tags = Array.from(set);
        });
      } catch (e) { console.error("[lorekeeper] frontmatter tags failed", e); }
    }
    if (plan.links && plan.links.length) {
      try {
        const txt = await this.app.vault.read(file);
        if (!txt.includes("## Related")) {
          const block = `\n\n## Related\n` + plan.links.map((l) => `- [[${l}]]`).join("\n") + "\n";
          await this.app.vault.modify(file, txt + block);
        }
      } catch (e) { console.error("[lorekeeper] links append failed", e); }
    }
  }

  async createSmartNote(title: string, _gist: string, plan: Plan) {
    if (!validFolder(plan.folder)) { new Notice("Lorekeeper: couldn't determine a folder for this note."); return; }
    const folder = plan.folder;
    const filename = plan.filename || slug(title);
    const path = `${folder}/${filename}.md`;
    if (this.app.vault.getAbstractFileByPath(path)) { new Notice(`Already exists: ${path}`); return; }
    await this.ensureFolder(folder);

    const tmplFile = plan.template ? this.templateFile(plan.template) : null;
    const tp = this.getTemplater();
    let file: TFile | null = null;

    if (tmplFile && tp && typeof tp.create_new_note_from_template === "function") {
      try {
        const r = await tp.create_new_note_from_template(tmplFile, folder, filename, true);
        file = r instanceof TFile ? r : (this.app.vault.getAbstractFileByPath(path) as TFile | null);
      } catch (e) { console.error("[lorekeeper] templater failed, falling back", e); }
    }
    if (!file) {
      let content = "";
      if (tmplFile) { try { content = this.lightSubstitute(await this.app.vault.read(tmplFile), title); } catch (_) {} }
      if (!content.trim()) content = `# ${title}\n\n`;
      file = await this.app.vault.create(path, content);
      await this.app.workspace.getLeaf(false).openFile(file);
    }
    await this.applyTagsAndLinks(file, plan);
    new Notice(`Created ${path}`);
    void this.reindex();
  }

  async fileNote(file: TFile, plan: Plan) {
    if (!validFolder(plan.folder)) { new Notice("Lorekeeper: couldn't determine a folder for this note."); return; }
    const folder = plan.folder;
    await this.ensureFolder(folder);
    const target = `${folder}/${file.name}`;
    if (file.path !== target) {
      if (this.app.vault.getAbstractFileByPath(target)) { new Notice(`A note already exists at ${target}`); return; }
      try { await this.app.fileManager.renameFile(file, target); } catch (e) { new Notice(`Move failed: ${e}`); return; }
    }
    const moved = (this.app.vault.getAbstractFileByPath(target) as TFile) ?? file;
    await this.applyTagsAndLinks(moved, plan);
    new Notice(`Filed to ${target}`);
    void this.reindex();
  }

  async reindex(): Promise<void> {
    try {
      new Notice("Lorekeeper: indexing…");
      const res = await requestUrl({ url: `${this.baseUrl()}/reindex`, method: "POST", contentType: "application/json", body: "{}", throw: false });
      if (res.status >= 400) { new Notice(`Lorekeeper: reindex error ${res.status}`); return; }
      new Notice(`Lorekeeper: indexed ${res.json?.indexed ?? "?"} notes`);
    } catch (e) { new Notice(`Lorekeeper: reindex failed (${e})`); }
  }

  async loadSettings() { this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData()); }
  async saveSettings() { await this.saveData(this.settings); }
}

class ChatView extends ItemView {
  plugin: LorekeeperPlugin;
  constructor(leaf: WorkspaceLeaf, plugin: LorekeeperPlugin) { super(leaf); this.plugin = plugin; }
  getViewType() { return VIEW_TYPE; }
  getDisplayText() { return "Lorekeeper"; }
  getIcon() { return "bot"; }

  async onOpen() {
    const root = this.containerEl.children[1] as HTMLElement;
    root.empty();
    root.addClass("lorekeeper-chat");
    const log = root.createDiv({ cls: "lk-log" });
    const form = root.createDiv({ cls: "lk-form" });
    const input = form.createEl("textarea", { cls: "lk-input", attr: { rows: "2", placeholder: "Ask your knowledge graph…  (Enter to send, Shift+Enter = newline)" } });
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
      copyBtn.onclick = async () => { await navigator.clipboard.writeText(a.answer); new Notice("Copied"); };
      const text = m.createDiv({ cls: "lk-text lk-markdown" });
      await MarkdownRenderer.render(this.plugin.app, a.answer, text, "", this as Component);
      if (a.sources && a.sources.length) {
        const src = m.createDiv({ cls: "lk-sources" });
        src.createSpan({ cls: "lk-sources-label", text: "Sources" });
        const chips = src.createDiv({ cls: "lk-chips" });
        for (const s of a.sources) {
          const chip = chips.createEl("a", { cls: "lk-chip", text: s.title });
          chip.onclick = (e) => { e.preventDefault(); this.plugin.app.workspace.openLinkText(s.path, "", false); };
        }
      }
      log.scrollTop = log.scrollHeight;
    };
    const submit = async () => {
      const q = input.value.trim();
      if (!q) return;
      addUser(q); input.value = "";
      const thinking = log.createDiv({ cls: "lk-msg lk-bot lk-thinking" });
      thinking.createDiv({ cls: "lk-who", text: "Lorekeeper" });
      thinking.createDiv({ cls: "lk-text", text: "…thinking…" });
      log.scrollTop = log.scrollHeight;
      const ans = await this.plugin.ask(q);
      thinking.remove();
      await addBot(ans);
    };
    send.onclick = submit;
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } });
  }
  async onClose() {}
}

class SmartNoteModal extends Modal {
  plugin: LorekeeperPlugin;
  titleVal = "";
  gistVal = "";
  plan: Plan | null = null;
  constructor(app: App, plugin: LorekeeperPlugin) { super(app); this.plugin = plugin; }
  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("lk-modal");
    contentEl.createEl("h3", { text: "New smart note" });
    new Setting(contentEl).setName("Title").addText((t) => {
      t.setPlaceholder("e.g. Second law of thermodynamics");
      t.onChange((v) => (this.titleVal = v.trim()));
      t.inputEl.style.width = "100%";
      setTimeout(() => t.inputEl.focus(), 0);
    });
    const gist = contentEl.createEl("textarea", { cls: "lk-modal-gist", attr: { rows: "4", placeholder: "Optional: a sentence or two so it can pick the folder / template / tags accurately." } });
    gist.oninput = () => (this.gistVal = gist.value);
    const preview = contentEl.createDiv({ cls: "lk-plan" });
    const actions = contentEl.createDiv({ cls: "lk-modal-actions" });
    const planBtn = actions.createEl("button", { text: "Suggest", cls: "mod-cta" });
    const createBtn = actions.createEl("button", { text: "Create" });
    createBtn.disabled = true;
    planBtn.onclick = async () => {
      if (!this.titleVal) { new Notice("Enter a title."); return; }
      planBtn.disabled = true; planBtn.setText("Thinking…");
      this.plan = await this.plugin.plan(this.gistVal || this.titleVal, this.titleVal);
      planBtn.disabled = false; planBtn.setText("Suggest");
      if (this.plan) { renderPlanPreview(preview, this.plan); createBtn.disabled = false; }
    };
    createBtn.onclick = async () => {
      if (!this.plan) return;
      this.close();
      await this.plugin.createSmartNote(this.titleVal, this.gistVal, this.plan);
    };
  }
  onClose() { this.contentEl.empty(); }
}

class PlanConfirmModal extends Modal {
  constructor(app: App, private plan: Plan, private heading: string, private okText: string, private onConfirm: () => Promise<void>) { super(app); }
  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("lk-modal");
    contentEl.createEl("h3", { text: this.heading });
    renderPlanPreview(contentEl.createDiv({ cls: "lk-plan" }), this.plan);
    const actions = contentEl.createDiv({ cls: "lk-modal-actions" });
    const ok = actions.createEl("button", { text: this.okText, cls: "mod-cta" });
    const cancel = actions.createEl("button", { text: "Cancel" });
    ok.onclick = async () => { this.close(); await this.onConfirm(); };
    cancel.onclick = () => this.close();
  }
  onClose() { this.contentEl.empty(); }
}

class LoreSettingTab extends PluginSettingTab {
  plugin: LorekeeperPlugin;
  constructor(app: App, plugin: LorekeeperPlugin) { super(app, plugin); this.plugin = plugin; }
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
